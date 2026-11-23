"""The calibrations of the offsets block of the plate page, without a browser (m5 §18.6, #501).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1`` and a
typed field is its ``.value``. A click on a *disabled* button still calls its handler on the
server (measured on Panel 1.10.0 and 1.9.4), so the controller's refusals are the guard (FM-169)
and a test clicks a disabled button to prove it. The tool is a stub that records what it was
given and moves nothing; the controller's tests (``test_plate_page_calibrations.py``) hold the
run into the document of its own, and ``tests/test_plate_page_calibrations_demo.py`` the whole
chain on the demo stand. What this cannot show is a browser: the layout, and whether a newcomer
finds the button, are for the session of ``/ui-check`` (FM-161).
"""

from __future__ import annotations

import os
import shlex
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from smc.hardware import Microscope
from smc.hardware.capabilities import LightSource, ObjectiveTurret
from smc.hardware.errors import HardwareError
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.registry import GROUP
from smc.plugins.slots import (
    CameraCalibration,
    MeasuredOffset,
    ObjectiveOffsetsResult,
    OffsetWell,
    PlateResult,
    TargetMeasure,
    WallImages,
    WellFit,
)
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui import plate_page
from smc.ui.controller import PlatePage
from smc.ui.guide import (
    CALIBRATE_DRY_RUN,
    CALIBRATE_LEAD,
    CALIBRATE_LIGHT_OFF,
    CALIBRATE_NOT_KNOWN,
    NOT_CONNECTED,
    OFFSETS_NO_DOCUMENT,
    busy_note,
    calibrate_changed,
    calibrate_is_the_plates,
    calibrate_position_unknown,
    calibrate_replaces,
    calibration_failed,
)
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The date a document written by ``_seed_calibration`` carries (``T0``).
SEEDED = "2026-11-17T09:00:00+00:00"
ANOTHER = "Calibrate the camera at another objective"


# --- stub tools and a turret double ----------------------------------------------------


class StubOffsetsParams(PluginParams):
    targets: tuple[str, ...]
    calibrations: tuple[Path, ...]
    confirm_turns: bool = False


class StubOffsets:
    """``objective-offsets`` that is never run here: the block's first click only arms."""

    name = provides = "objective-offsets"
    summary = "A test tool."
    Params = StubOffsetsParams
    Result = ObjectiveOffsetsResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[StubOffsetsParams]] = []

    def run(
        self, microscope: Any, params: StubOffsetsParams, ctx: RunContext
    ) -> ObjectiveOffsetsResult:
        StubOffsets.seen.append(params)
        fit = WellFit(
            well="A1",
            recorded_xy_um=(0.0, 0.0),
            center_xy_um=(0.0, 0.0),
            diameter_um=6000.0,
            size_error=0.0,
            rms_um=0.1,
            n_points=100,
            walls=(),
        )
        measure = TargetMeasure(
            objective=params.targets[0],
            calibration="cal.json",
            calibration_measured="2026-11-10T10:00:00+00:00",
            pixel_size_um=2.5,
            z_after_turn_um=None,
            wells=(
                OffsetWell(
                    well="A1", origin_xy_um=(0.0, 0.0), delta_xy_um=(1.0, 2.0), fit=fit
                ),
            ),
            shift_xy_um=(1.0, 2.0),
            spread_um=0.03,
            max_spread_um=2.0,
            frames=24,
            folder=None,
        )
        return ObjectiveOffsetsResult(
            plate_objective=TEN,
            reference=TEN,
            reference_in_profile=False,
            measures=(measure,),
            offsets=(
                MeasuredOffset(objective=params.targets[0], xy_offset_um=(1.0, 2.0)),
            ),
            objective_in_place=params.targets[0],
            elapsed_s=1.0,
        )


class StubCalibrationParams(PluginParams):
    """The real tool's three fields, so that the form offers the advanced one too."""

    step_um: float | None = None
    settle_ms: float = 0.0
    from_run_doc: Path | None = None


class StubCalibration:
    """``camera-calibration`` without a stage: the calibration of the turret's current label, or the failure the test sets."""

    name = provides = "camera-calibration"
    summary = "A test tool."
    Params = StubCalibrationParams
    Result = CameraCalibration
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    seen: ClassVar[list[StubCalibrationParams]] = []
    fails: ClassVar[str | None] = None

    def run(
        self, microscope: Any, params: StubCalibrationParams, ctx: RunContext
    ) -> CameraCalibration:
        StubCalibration.seen.append(params)
        if StubCalibration.fails is not None:
            raise PluginRunError("camera-calibration", StubCalibration.fails)
        return _calibration(microscope.require(ObjectiveTurret).current())


class SlowParams(PluginParams):
    pass


class SlowResult(PluginResult):
    pass


class Slow:
    """A tool that waits until the test lets it go: an activity that is going."""

    name = provides = "slow"
    summary = "A test tool that waits."
    Params = SlowParams
    Result = SlowResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> SlowResult:
        Slow.started.set()
        Slow.release.wait(10.0)
        return SlowResult()


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    StubOffsets.seen = []
    StubCalibration.seen = []
    StubCalibration.fails = None
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()


class Light:
    """A light source the test sets: its switch reads on, off or nothing smc knows (m5 §18.6 D20)."""

    def __init__(self, reading: bool | None) -> None:
        self.reading = reading

    def device(self) -> str:
        return "LED"

    def intensity_knob(self) -> Any:
        return None

    def intensity_native(self) -> float | None:
        return None

    def set_intensity_native(self, value: float) -> float:
        return value

    def is_on(self) -> bool | None:
        return self.reading

    def set_on(self, on: bool) -> bool:
        self.reading = on
        return on


class Turret:
    """A turret the test sets: the label in place and the positions' labels."""

    def __init__(self, labels: list[str], now: str | None) -> None:
        self.labels_ = labels
        self.now = now

    def labels(self) -> list[str]:
        return list(self.labels_)

    def current(self) -> str:
        if self.now is None:
            raise HardwareError("the turret is between two positions")
        return self.now

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.now = label
        return label

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0

    def offset(self, label: str) -> Any:
        raise NotImplementedError


# --- records written out by hand -------------------------------------------------------


def _calibration(
    objective: str | None, assumed: tuple[str, ...] = ()
) -> CameraCalibration:
    return CameraCalibration(
        objective=objective,
        image_shape_px=(384, 512),
        matrix_um_per_px=((0.5, 0.0), (0.0, -0.5)),
        pixel_size_um=0.5,
        rotation_deg=0.0,
        mirrored=True,
        residual_um=0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=48.0,
        assumed=assumed,
    )


def _write(
    path: Path,
    plugin: str,
    provides: str,
    result: dict[str, Any] | None,
    *,
    params: dict[str, Any] | None = None,
    minute: int = 0,
) -> None:
    then = T0 + timedelta(minutes=minute)
    write_stage(
        path,
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
            params=params if params is not None else {},
            result=result,
        ),
    )


def _fit(well: str, centre: tuple[float, float]) -> WellFit:
    return WellFit(
        well=well,
        recorded_xy_um=centre,
        center_xy_um=centre,
        diameter_um=6000.0,
        size_error=0.0,
        rms_um=0.1,
        n_points=100,
        walls=(),
    )


def _seed_plate(
    run_doc: Path, *, by_eye: bool = False, minute: int | None = None
) -> None:
    """A plate registered under the 10X from its walls, or by eye (no block)."""
    walls = WallImages(
        objective=TEN,
        pixel_size_um=1.1,
        image_shape_px=(100, 100),
        search_um=1000.0,
        wells=(_fit("A1", (0.0, 0.0)), _fit("H12", (99000.0, -63000.0))),
        frames=8,
        folder=None,
        elapsed_s=10.0,
    )
    plate = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="references" if by_eye else "wall-images",
        residual_um=1.0,
        scale_error=None,
        n_refs=2,
        walls=None if by_eye else walls,
        objective=TEN,
    )
    _write(
        run_doc,
        "plate-registration" if by_eye else "plate-walls",
        "plate",
        plate.model_dump(mode="json"),
        params={"plate": plate.plate},
        # A registration by eye is a later attempt: the latest provider decides the slot.
        minute=(1 if by_eye else 0) if minute is None else minute,
    )


def _hold(
    run_doc: Path, position: int, objective: str, assumed: tuple[str, ...] = ()
) -> Path:
    """The document a calibration at ``objective`` would leave for turret ``position``."""
    path = run_doc.parent / "calibrations" / f"position-{position}.json"
    path.parent.mkdir(exist_ok=True)
    _write(
        path,
        "camera-calibration",
        "camera-calibration",
        _calibration(objective, assumed).model_dump(mode="json"),
    )
    return path


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(name=c.name, value=f"{__name__}:{c.__name__}", group=GROUP)
        for c in (StubCalibration, StubOffsets, Slow)
    ]


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_view(
    fake_microscope: Microscope, run_doc: Path
) -> Iterator[Callable[..., PlatePageView]]:
    """A view over the fake stand (or the one given) with the stub tools; every page it built is closed."""
    pages: list[PlatePage] = []

    def make(microscope: Microscope | None = None) -> PlatePageView:
        page = PlatePage(
            microscope or fake_microscope, run_doc, entry_points=_entry_points()
        )
        pages.append(page)
        view = PlatePageView(page)
        view.show_step("plate")
        view.refresh()
        view.tick()
        return view

    try:
        yield make
    finally:
        Slow.release.set()
        for page in pages:
            page.close()


def _click(button: Any) -> None:
    button.clicks += 1


def _turn(microscope: Microscope, label: str) -> None:
    """The operator's turn of the turret: what the page does not do, and these tests do not test."""
    microscope.require(ObjectiveTurret).select(label, confirm=True)


def _objects(items: list[Any]) -> list[Any]:
    found: list[Any] = []
    for item in items:
        found.extend(item.select())
    return found


def _ready(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    label: str = FORTY,
) -> PlatePageView:
    """A view on a plate registered from its walls with the 10X, the turret turned to ``label``, drawn."""
    _seed_plate(run_doc)
    _turn(fake_microscope, label)
    view = make_view()
    view.refresh()
    view.tick()
    return view


# --- the button: what it names and what it waits for (D12) ------------------------------


def test_the_calibrate_button_names_the_objective_in_place_and_waits_for_each_reason_beside_it(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc)

    # An objective that is not the plate's, nothing running: it names it, enabled, with no wait.
    assert view.offsets_box.visible is True
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert view.offsets_calibrate_button.disabled is False
    assert view.offsets_calibrate_wait.object == ""
    assert view.offsets_calibrate_wait.visible is False
    assert view._calibrate_for == FORTY
    assert view.calibrate_lead_text.object == CALIBRATE_LEAD

    # An activity goes: the first reason, the tool's busy note, and the objective still named.
    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()
    assert view.offsets_calibrate_button.disabled is True
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert view.offsets_calibrate_wait.object == busy_note("slow", live=False)
    Slow.release.set()
    assert view.page.join(10.0)
    view.tick()
    assert view.offsets_calibrate_button.disabled is False

    # The plate's own objective in place: the button names none and says why.
    _turn(fake_microscope, TEN)
    view.tick()
    assert view.offsets_calibrate_button.label == ANOTHER
    assert view.offsets_calibrate_button.disabled is True
    assert view.offsets_calibrate_wait.object == calibrate_is_the_plates(TEN)
    assert view._calibrate_for is None

    # An objective not known (a turret between two positions): the same label, the other reason.
    labels = fake_microscope.require(ObjectiveTurret).labels()
    fake_microscope.override(ObjectiveTurret, Turret(labels, None))
    view.tick()
    assert view.offsets_calibrate_button.label == ANOTHER
    assert view.offsets_calibrate_button.disabled is True
    assert view.offsets_calibrate_wait.object == CALIBRATE_NOT_KNOWN
    assert view._calibrate_for is None

    # A dry-run page that holds a real registration, and a closed stand.
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        dry_view = make_view(dry)
        assert dry_view.offsets_calibrate_button.disabled is True
        assert dry_view.offsets_calibrate_wait.object == CALIBRATE_DRY_RUN
    finally:
        dry.close()
    closed = PlatePage.closed(
        fake_microscope.profile, run_doc, entry_points=_entry_points()
    )
    try:
        closed_view = PlatePageView(closed)
        closed_view.show_step("plate")
        closed_view.refresh()
        closed_view.tick()
        assert closed_view.offsets_box.visible is True
        assert closed_view.offsets_calibrate_button.disabled is True
        assert closed_view.offsets_calibrate_wait.object == NOT_CONNECTED
    finally:
        closed.close()


def test_the_blocks_button_waits_with_the_light_off(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-206 (D20): a calibration that the tool refuses for a dark frame replaces the position's
    # usable document, so the button waits for the light, with the reason beside it.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    light = Light(False)
    fake_microscope.override(LightSource, light)
    view.tick()

    assert view.offsets_calibrate_button.disabled is True
    assert view.offsets_calibrate_wait.object == CALIBRATE_LIGHT_OFF
    # It still names the objective in place, and the click would carry it.
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert view._calibrate_for == FORTY

    # The light on, or one nobody can read: nothing to wait for (an unknown refuses nothing).
    for reading in (True, None):
        light.reading = reading
        view.tick()
        assert view.offsets_calibrate_button.disabled is False
        assert view.offsets_calibrate_wait.object == ""

    # The other reasons come first, as the pure function orders them: the plate's own objective.
    light.reading = False
    _turn(fake_microscope, TEN)
    view.tick()
    assert view.offsets_calibrate_wait.object == calibrate_is_the_plates(TEN)


def test_a_page_that_does_not_offer_the_block_says_nothing_of_the_calibrations(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # The dry-run states of `/ui-check` read `summary.txt` for the block's words: a hidden block
    # must carry none (the page's panes are scraped whether they are shown or not).
    _hold(run_doc, 1, FORTY)
    _turn(fake_microscope, FORTY)
    for by_eye in (None, True):
        if by_eye:
            _seed_plate(run_doc, by_eye=True)
        view = make_view()
        assert view.offsets_box.visible is False
        assert view.page.view().calibrations == ()
        for pane in (
            view.calibrations_held_text,
            view.calibrate_lead_text,
            view.offsets_calibrate_wait,
        ):
            assert pane.object == ""
        assert view._calibrate_for is None
    # The block, offered, says them; the plate registered again by eye takes them back.
    _seed_plate(run_doc, minute=2)
    view.refresh()
    view.tick()
    assert view.calibrate_lead_text.object == CALIBRATE_LEAD
    assert view.calibrations_held_text.object != ""
    _seed_plate(run_doc, by_eye=True, minute=3)
    view.refresh()
    view.tick()
    assert view.offsets_box.visible is False
    assert view.calibrate_lead_text.object == ""
    assert view.calibrations_held_text.object == ""
    assert view.offsets_calibrate_wait.object == ""
    assert view._calibrate_for is None


def test_a_dry_run_page_on_a_walls_plate_shows_the_wait_beside_the_calibrate_button(
    make_view: Callable[..., PlatePageView],
    fake_core: FakeCore,
    run_doc: Path,
) -> None:
    # FM-182: the block walked on a dry-run page that holds a real registration; the click
    # arrives all the same, and the controller's refusal is the outcome, with nothing written.
    _seed_plate(run_doc)
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        view = make_view(dry)
        before = run_doc.read_bytes()

        assert view.offsets_calibrate_button.disabled is True
        assert view.offsets_calibrate_wait.object == CALIBRATE_DRY_RUN
        _click(view.offsets_calibrate_button)
        outcome = view.page.status().outcome
        folder = run_doc.parent / "calibrations"

        assert outcome == f"Camera calibration: {CALIBRATE_DRY_RUN}"
        assert StubCalibration.seen == []
        assert not folder.exists()
        assert run_doc.read_bytes() == before
    finally:
        dry.close()


def test_the_calibrate_click_sends_the_label_the_button_showed_and_the_forms_typed_fields(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc)
    sent: list[tuple[str, tuple[str, ...]]] = []

    def spy(expected: str, pairs: Any = ()) -> None:
        sent.append((expected, tuple(pairs)))

    monkeypatch.setattr(view.page, "calibrate_objective", spy)
    # The expert form's own fields, typed: what the Stand screen's button sends too.
    view.tool_fields["camera-calibration"]["step_um"].value = "40"
    view.tool_fields["camera-calibration"]["settle_ms"].value = "250"
    # Not the form's advanced `from_run_doc`: a reuse that the tool refuses would already have
    # replaced the position's usable document with a failed record (the review of #501).
    view.tool_fields["camera-calibration"]["from_run_doc"].value = "cal-10x.json"

    _click(view.offsets_calibrate_button)

    assert sent == [(FORTY, ("step_um=40", "settle_ms=250"))]


def test_a_click_on_a_stale_calibrate_button_calibrates_nothing(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-177: the turret changes with no tick in between, then the click. A test that ticks
    # proves nothing: the label is the one drawn up to a second ago.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    folder = run_doc.parent / "calibrations"
    before = run_doc.read_bytes()

    for shown, now in ((FORTY, TWENTY), (TWENTY, FORTY)):
        _turn(fake_microscope, shown)
        view.tick()
        assert view.offsets_calibrate_button.label == f"Calibrate the camera at {shown}"
        _turn(fake_microscope, now)
        # The stale label is asserted first: nothing redrew it.
        assert view.offsets_calibrate_button.label == f"Calibrate the camera at {shown}"
        assert view._calibrate_for == shown

        _click(view.offsets_calibrate_button)

        assert view.page.status().outcome == (
            f"Camera calibration: {calibrate_changed(shown, now)}"
        )
        assert view.page.status().running is None
        assert StubCalibration.seen == []
        assert not folder.exists()
        assert run_doc.read_bytes() == before
        # Nothing in the click turned the turret (FM-14).
        assert fake_microscope.require(ObjectiveTurret).current() == now


def test_a_click_on_a_disabled_calibrate_button_runs_nothing(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-169: the handler runs although the button is disabled, and the controller refuses.
    view = _ready(make_view, fake_microscope, run_doc)
    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()
    assert view.offsets_calibrate_button.disabled is True
    before = run_doc.read_bytes()

    _click(view.offsets_calibrate_button)

    assert view.message.object == "slow is running"
    assert StubCalibration.seen == []
    assert not (run_doc.parent / "calibrations").exists()
    assert run_doc.read_bytes() == before
    Slow.release.set()
    assert view.page.join(10.0)


def test_the_again_button_says_what_it_replaces_before_the_click(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-172: a new calibration replaces the position's document, and a failed one leaves none.
    _hold(run_doc, 1, FORTY)
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    assert (
        view.offsets_calibrate_button.label == f"Calibrate the camera again at {FORTY}"
    )
    assert view.offsets_calibrate_wait.object == calibrate_replaces(FORTY, SEEDED)
    assert view.offsets_calibrate_wait.visible is True
    assert view.offsets_calibrate_button.disabled is False

    # The 20X has no document: no `again`, and nothing to say before the click.
    _turn(fake_microscope, TWENTY)
    view.tick()
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {TWENTY}"
    assert view.offsets_calibrate_wait.object == ""
    assert view.offsets_calibrate_wait.visible is False


def test_a_position_that_holds_another_objectives_calibration_says_whose_it_replaces(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-216, found by the review of #501: position 1 holds the 20X's calibration (made when the
    # 20X sat there; the turret has been relabelled since) and the 40X is now at position 1. A
    # new calibration replaces that document, so the line names the objective that loses its
    # calibration, and the button does not say `again` for an objective that has none.
    _hold(run_doc, 1, TWENTY)
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert view.offsets_calibrate_wait.object == calibrate_replaces(TWENTY, SEEDED)
    assert FORTY not in view.offsets_calibrate_wait.object
    # The list says the same before the click: position 1's document holds the 20X's calibration
    # and is not offered as that position's (D21), so the 40X has none and neither does the 20X.
    lines = view.calibrations_held_text.object.splitlines()
    assert lines[1] == (
        f"position 1 ({FORTY}): the document holds a calibration of {TWENTY}, not of "
        "the objective at this position now (the turret was turned during a run, or "
        f"relabelled since): calibrate {FORTY} again, or type the documents to use"
    )
    assert f"{TWENTY}: calibrated on 2026-11-17" not in lines
    assert f"{FORTY}: no calibration yet" in lines
    assert f"{TWENTY}: no calibration yet" in lines


def test_a_calibration_from_the_page_is_listed_with_its_objective_and_date(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    before = run_doc.read_bytes()

    _click(view.offsets_calibrate_button)
    assert view.page.join(10.0)
    view.tick()

    folder = run_doc.parent / "calibrations"
    lines = view.calibrations_held_text.object.splitlines()
    assert lines[0] == (
        f"Camera calibrations the page holds, one document per objective, in {folder}:"
    )
    assert lines[1].startswith(f"{FORTY}: calibrated on ")
    assert (folder / "position-1.json").is_file()
    # The registration is the one it was: the button wrote a document of its own.
    assert run_doc.read_bytes() == before
    assert view.page.view().registered is not None
    # The label now says `again`, with what a new one replaces.
    assert (
        view.offsets_calibrate_button.label == f"Calibrate the camera again at {FORTY}"
    )
    assert view.page.status().outcome is not None
    assert view.page.status().outcome.startswith("Camera calibration done in ")  # type: ignore[union-attr]


# --- the list (D15) -----------------------------------------------------------------


def test_the_list_names_each_document_its_objective_and_date_each_objective_with_none_and_why_one_is_left_out(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    _hold(run_doc, 1, FORTY)
    _hold(run_doc, 2, TEN)  # the plate's own objective
    _hold(run_doc, 5, "Nikon 60X Plan Apo")  # not on the turret now
    broken = _hold(run_doc, 3, "Objective-2")
    broken.write_text("{not json", encoding="utf-8")
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    folder = run_doc.parent / "calibrations"
    lines = view.calibrations_held_text.object.splitlines()
    problem = lines[3]
    assert lines == [
        f"Camera calibrations the page holds, one document per objective, in {folder}:",
        f"{FORTY}: calibrated on 2026-11-17",
        f"{TEN}: calibrated on 2026-11-17; the plate's objective, not measured from itself",
        problem,
        # Position 5 holds an objective that is no label of the turret now: not its position's.
        "position 5 (Objective-4): the document holds a calibration of Nikon 60X Plan "
        "Apo, not of the objective at this position now (the turret was turned during "
        "a run, or relabelled since): calibrate Objective-4 again, or type the "
        "documents to use",
        # Each other objective of the turret that no usable document names, in turret order.
        "Objective-2: no calibration yet",
        f"{TWENTY}: no calibration yet",
        "Objective-4: no calibration yet",
        "Objective-5: no calibration yet",
    ]
    # A document that cannot be read says so at its position, with the position's own label.
    assert problem.startswith("position 3 (Objective-2): ")
    assert "is not valid JSON" in problem
    assert view.calibrations_held_text.visible is True


def test_the_list_says_a_document_in_another_objectives_file_and_lists_that_objective_as_missing(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # Acceptance 3 (D21): the 10X's calibration, run into position-1.json (the 40X's file), was
    # listed as the 10X's with no warning and the 40X as missing. Its line now says whose it is
    # and where it is wrong; the 40X has none; a document at its own position is as it was; and
    # the notes of a document that is not offered are not shown under it.
    _hold(run_doc, 1, TEN, assumed=("a note of the 10X's calibration",))
    _hold(run_doc, 4, TWENTY, assumed=("a note of the 20X's calibration",))
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    folder = run_doc.parent / "calibrations"
    lines = view.calibrations_held_text.object.splitlines()
    assert lines == [
        f"Camera calibrations the page holds, one document per objective, in {folder}:",
        f"position 1 ({FORTY}): the document holds a calibration of {TEN}, not of the "
        "objective at this position now (the turret was turned during a run, or "
        f"relabelled since): calibrate {FORTY} again, or type the documents to use",
        f"{TWENTY}: calibrated on 2026-11-17",
        f"  {TWENTY}: a note of the 20X's calibration",
        # In turret order: the 40X has none, whatever position 1's file holds.
        f"{FORTY}: no calibration yet",
        "Objective-2: no calibration yet",
        "Objective-4: no calibration yet",
        "Objective-5: no calibration yet",
    ]
    # The button, for the 40X, says what the next calibration replaces: the 10X's document.
    assert view.offsets_calibrate_wait.object == calibrate_replaces(TEN, SEEDED)


def test_with_the_labels_unread_a_document_is_listed_as_it_is_and_nothing_is_said_of_its_position(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # Labels that cannot be read say nothing of a position: the line is the usable one, and no
    # objective is listed as missing (FM-145: the page's copy of the labels is all it has).
    _hold(run_doc, 1, TWENTY)
    _seed_plate(run_doc)
    fake_microscope.override(ObjectiveTurret, Turret([], FORTY))
    view = make_view()
    view.refresh()
    view.tick()

    lines = view.calibrations_held_text.object.splitlines()
    assert lines[1:] == [f"{TWENTY}: calibrated on 2026-11-17"]


def test_the_notes_of_a_calibration_are_listed_under_its_line(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    _hold(run_doc, 1, FORTY, assumed=("step of 50 µm: no pixel size was known", "two"))
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    lines = view.calibrations_held_text.object.splitlines()
    assert lines[1:4] == [
        f"{FORTY}: calibrated on 2026-11-17",
        f"  {FORTY}: step of 50 µm: no pixel size was known",
        f"  {FORTY}: two",
    ]


def test_a_failed_calibration_is_in_the_list_in_the_tools_words(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    words = "the frame of Nikon 40X is dark or flat: check the light path"
    StubCalibration.fails = words

    _click(view.offsets_calibrate_button)
    assert view.page.join(10.0)
    view.tick()

    lines = view.calibrations_held_text.object.splitlines()
    assert lines[1] == f"position 1 ({FORTY}): {calibration_failed(words)}"
    # A failed record leaves the position with no usable document: no `again`, and nothing left
    # to replace, so nothing is said beside the button (FM-172).
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert view.offsets_calibrate_wait.object == ""
    assert view.offsets_calibrate_wait.visible is False
    assert f"{FORTY}: no calibration yet" in lines


def test_the_live_view_is_the_activity_the_calibrate_button_waits_for(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    view.page.live(True)
    try:
        view.tick()
        assert view.offsets_calibrate_button.disabled is True
        assert view.offsets_calibrate_wait.object == busy_note("live", live=True)
    finally:
        view.page.live(False)
        assert view.page.join(10.0)


def test_the_list_is_text_and_not_markup(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-139: a label, a path and a tool's words come from outside.
    label = "40X [bold]*lens*\nnext <b>x</b>"
    labels = fake_microscope.require(ObjectiveTurret).labels()
    labels[0] = label
    fake_microscope.override(ObjectiveTurret, Turret(labels, label))
    _hold(run_doc, 1, label, assumed=("a [link](x)\nsecond line",))
    view = _ready(make_view, fake_microscope, run_doc, label)

    text = view.calibrations_held_text.object
    assert type(view.calibrations_held_text) is pn.pane.Str
    assert type(view.offsets_calibrate_wait) is pn.pane.Str
    assert type(view.calibrate_lead_text) is pn.pane.Str
    # The label is on one line, with its markup as the words it is.
    assert "40X [bold]*lens* next <b>x</b>: calibrated on 2026-11-17" in text
    assert "  40X [bold]*lens* next <b>x</b>: a [link](x) second line" in text
    assert text.count("\n") == len(text.splitlines()) - 1
    assert "\n" not in view.offsets_calibrate_button.label
    assert view.offsets_calibrate_button.label == (
        "Calibrate the camera again at 40X [bold]*lens* next <b>x</b>"
    )


# --- the field (D16) ----------------------------------------------------------------


def test_the_lines_under_the_field_are_the_typed_ones_and_empty_for_an_empty_field(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
) -> None:
    _hold(run_doc, 1, FORTY)
    _hold(run_doc, 4, TWENTY)
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    elsewhere = tmp_path / "elsewhere.json"
    _write(
        elsewhere,
        "camera-calibration",
        "camera-calibration",
        _calibration(TWENTY).model_dump(mode="json"),
    )

    # An empty field: the measure arms the held documents, and nothing is said under the field.
    _click(view.offsets_measure_button)
    assert view._offsets_armed == (FORTY, TWENTY)
    assert view.calibration_reads_text.object == ""
    assert view.calibration_reads_text.visible is False
    assert view.offsets_wait_text.object == ""
    assert view.offsets_confirm_box.visible is True
    # Nothing turned and nothing ran: the question is asked and the turret reads the 40X.
    assert StubOffsets.seen == []
    assert fake_microscope.require(ObjectiveTurret).current() == FORTY

    # A typed field is used, with one line under it for the typed path alone.
    view.calibrations_input.value = str(elsewhere)
    assert view.calibration_reads_text.object == (
        f"{elsewhere}: calibrated at {TWENTY} on 2026-11-17"
    )
    _click(view.offsets_measure_button)
    assert view._offsets_armed == (TWENTY,)

    # Emptied again: the lines go, the documents of the page are the default once more.
    view.calibrations_input.value = ""
    assert view.calibration_reads_text.object == ""
    assert view.calibration_reads_text.visible is False


def test_with_nothing_held_and_nothing_typed_the_wait_says_how_to_make_one(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    _click(view.offsets_measure_button)

    assert view._offsets_armed is None
    assert view.offsets_wait_text.object == OFFSETS_NO_DOCUMENT
    assert "the button above" in OFFSETS_NO_DOCUMENT


def _deny(monkeypatch: pytest.MonkeyPatch, state: dict[str, bool], *names: str) -> None:
    """``Path.stat`` raises ``PermissionError`` for the calibrations folder's files named in ``names`` (every one when none is) while ``state["on"]``: a folder the account may not search, on any platform."""
    real = Path.stat

    def stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if (
            state["on"]
            and self.parent.name == "calibrations"
            and (not names or self.name in names)
        ):
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_a_denied_folder_is_one_line_under_the_heading_and_no_objective_is_listed_missing(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D22: twelve lines for one folder the account cannot search, and `no calibration yet` under
    # them for every objective, said that nothing was calibrated when the page did not know.
    _hold(run_doc, 1, FORTY)
    _deny(monkeypatch, {"on": True})
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    folder = run_doc.parent / "calibrations"
    denied = (
        f"{folder}: the page cannot look into this folder (PermissionError: "
        "Permission denied); which objectives have a calibration is not known"
    )
    assert view.calibrations_held_text.object.splitlines() == [
        f"Camera calibrations the page holds, one document per objective, in {folder}:",
        denied,
    ]
    assert "no calibration yet" not in view.calibrations_held_text.object
    assert view.calibrations_held_text.visible is True
    assert type(view.calibrations_held_text) is pn.pane.Str


def test_a_denied_folder_wait_is_the_line_and_is_taken_back_when_the_folder_opens(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-247: the wait beside *Measure the offsets* was read for the documents (and the denial)
    # that were there at the click; once either changes it must not be drawn as the truth.
    state = {"on": True}
    _hold(run_doc, 1, FORTY)
    _deny(monkeypatch, state)
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    denied = view.page.view().calibrations_denied
    assert denied is not None

    _click(view.offsets_measure_button)
    assert view._offsets_armed is None
    assert view.offsets_wait_text.object == denied
    assert view.offsets_wait_text.object != OFFSETS_NO_DOCUMENT

    # The folder is opened to the account: the redraw reads it, the line is gone with the wait.
    state["on"] = False
    view.refresh()
    view.tick()
    assert view.offsets_wait_text.object == ""
    lines = view.calibrations_held_text.object.splitlines()
    assert lines[1] == f"{FORTY}: calibrated on 2026-11-17"
    assert denied not in lines
    # Its absence is a fact again: the objectives with no document are listed.
    assert f"{TWENTY}: no calibration yet" in lines


def test_a_denial_alone_changing_takes_the_wait_of_the_documents_back(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-247: the documents held do not change (there are none), only the denial does: the wait
    # drawn for the folder as it was (`OFFSETS_NO_DOCUMENT`) is dropped with it, and the next
    # click says the folder's line.
    state = {"on": False}
    _deny(monkeypatch, state, "position-2.json")
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    _click(view.offsets_measure_button)
    assert view.offsets_wait_text.object == OFFSETS_NO_DOCUMENT
    assert view.page.view().calibrations == ()

    state["on"] = True
    view.refresh()
    view.tick()

    assert view.page.view().calibrations == ()
    assert view.offsets_wait_text.object == ""
    folder = run_doc.parent / "calibrations"
    assert view.calibrations_held_text.object.splitlines() == [
        f"Camera calibrations the page holds, one document per objective, in {folder}:",
        f"{folder}: the page cannot look into this folder (PermissionError: "
        "Permission denied); which objectives have a calibration is not known",
    ]
    _click(view.offsets_measure_button)
    assert view.offsets_wait_text.object == view.page.view().calibrations_denied


def test_a_calibration_made_after_the_wait_was_drawn_takes_the_stale_wait_back(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # The review of #501: the wait beside *Measure the offsets* is read at a click; once the page
    # has made the calibration it asked for, it must not go on saying that none is held.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    _click(view.offsets_measure_button)
    assert view.offsets_wait_text.object == OFFSETS_NO_DOCUMENT

    _click(view.offsets_calibrate_button)
    assert view.page.join(10.0)
    view.tick()

    assert view.calibrations_held_text.object.splitlines()[1].startswith(
        f"{FORTY}: calibrated on "
    )
    assert view.offsets_wait_text.object == ""
    assert view.offsets_wait_text.visible is False
    # The next click reads again, and says the objective that is still to calibrate is not
    # a document: the 40X alone is held and is a target, so Measure now arms it.
    _click(view.offsets_measure_button)
    assert view._offsets_armed == (FORTY,)


def test_a_closed_stand_lists_the_documents_it_holds_without_the_turrets_labels(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-98, FM-125: a page reopened with documents from an earlier session and the stand closed has
    # no labels (a closed stand has no turret); the block is offered and its list is drawn, and a
    # document on a position past the labels read is no exception.
    _seed_plate(run_doc)
    _hold(run_doc, 1, FORTY)
    _hold(run_doc, 9, "Nikon 60X Plan Apo")
    closed = PlatePage.closed(
        fake_microscope.profile, run_doc, entry_points=_entry_points()
    )
    try:
        view = PlatePageView(closed)
        view.show_step("plate")
        view.refresh()
        view.tick()
        lines = view.calibrations_held_text.object.splitlines()
        assert view.offsets_box.visible is True
    finally:
        closed.close()

    # Whether an objective is on the turret is not said: the turret is not read. Nothing is
    # listed as missing either, since the page does not know the turret's other objectives.
    assert lines[1:] == [
        f"{FORTY}: calibrated on 2026-11-17",
        "Nikon 60X Plan Apo: calibrated on 2026-11-17",
    ]


# --- where the widgets are, and that every handler is guarded (FM-171, FM-202) -------------


def test_the_new_widgets_are_in_the_block_once_and_the_guard_test_walks_the_new_handler(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    view = make_view()
    expert = _objects(view.screens["expert"])
    plate = _objects(view.screens["plate"])
    new = (
        view.calibrations_held_text,
        view.calibrate_lead_text,
        view.offsets_calibrate_button,
        view.offsets_calibrate_wait,
    )
    for widget in new:
        assert plate.count(widget) == 1
        assert expert.count(widget) == 1
    # In the block, between the objective in place and the field.
    objects = list(view.offsets_box.objects)
    assert objects[
        objects.index(view.objective_in_place_text) : objects.index(
            view.calibrations_input
        )
    ] == [view.objective_in_place_text, *new]
    assert not any(isinstance(o, plate_page.figure) for o in view.offsets_box.select())
    # The button's text is its `label=`, and its click goes through the page's guard.
    assert view.offsets_calibrate_button.label == ANOTHER
    ours = [
        watcher.fn
        for by_name in view.offsets_calibrate_button.param.watchers.values()
        for watchers in by_name.values()
        for watcher in watchers
        if getattr(watcher.fn, "__module__", "") == plate_page.__name__
    ]
    assert ours
    assert all(hasattr(fn, "__wrapped__") for fn in ours)
    # The field's label is a sentence of the design: it wraps and is not cut at the panel's edge
    # (the browser session of #501 saw it run past it at 1280 x 800).
    assert any(
        "white-space: normal" in str(sheet)
        for sheet in view.calibrations_input.stylesheets
    )


def test_a_second_tab_draws_its_own_button_objective(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-151: the objective a button names is the view's, and so is the click's label.
    first = _ready(make_view, fake_microscope, run_doc, FORTY)
    _turn(fake_microscope, TWENTY)
    second = PlatePageView(first.page)
    second.show_step("plate")
    second.refresh()
    second.tick()

    assert first._calibrate_for == FORTY
    assert second._calibrate_for == TWENTY
    assert first.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert second.offsets_calibrate_button.label == f"Calibrate the camera at {TWENTY}"
    # The first tab's click is the stale one: the controller refuses it, and says both labels.
    _click(first.offsets_calibrate_button)
    assert first.page.status().outcome == (
        f"Camera calibration: {calibrate_changed(FORTY, TWENTY)}"
    )
    assert StubCalibration.seen == []
    # The second tab's names what is in place: it runs.
    _click(second.offsets_calibrate_button)
    assert first.page.join(10.0)
    assert [p.step_um for p in StubCalibration.seen] == [None]


def test_the_tick_reads_no_document_and_no_stand_for_the_calibrate_button(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-145: the label and the wait come from the status and the last view.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    calls: list[str] = []
    for name in (
        "view",
        "_held_calibrations",
        "offsets_step",
        "_halted",
        "objective_labels",
        "_turret_label",
        "_turret_reading",
    ):
        monkeypatch.setattr(view.page, name, lambda *a, _n=name: calls.append(_n) or ())
    for label in (TWENTY, FORTY, TEN):
        _turn(fake_microscope, label)
        view.tick()

    assert calls == []
    assert view.offsets_calibrate_button.label == ANOTHER


def test_the_tick_makes_no_read_of_the_turret_of_its_own_for_the_calibrate_button(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-145: the status reads the turret once at a tick; the button's draw adds none of its
    # own (a drawer that read `labels()` or `current()` would take the core's lock at every tick
    # of every tab, while a scan runs). Counted on a turret double, whatever route the read took.
    reads: list[str] = []

    class Counting(Turret):
        def labels(self) -> list[str]:
            reads.append("labels")
            return super().labels()

        def current(self) -> str:
            reads.append("current")
            return super().current()

    labels = fake_microscope.require(ObjectiveTurret).labels()
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    fake_microscope.override(ObjectiveTurret, Counting(labels, FORTY))
    view.tick()
    baseline = list(reads)
    for label in (TWENTY, FORTY, TEN):
        fake_microscope.require(ObjectiveTurret).select(label, confirm=True)
        reads.clear()
        view.tick()
        assert reads == baseline, reads
    assert "labels" not in baseline
    assert baseline.count("current") <= 1
    assert view.offsets_calibrate_button.label == ANOTHER


# --- the Stand screen's button keeps the plate's calibration (D19, FM-216) -------------


def _routed(plate: str, objective: str, position: int) -> str:
    """The Stand screen's note of where the calibration goes, as m5 §18.6.3 words it."""
    return (
        f"The plate was registered with {plate}: a calibration at {objective} here is "
        f"kept in its own document, calibrations/position-{position}.json, and the "
        "plate's calibration stays as it is. Measure the offsets between objectives "
        "(Plate screen) uses it."
    )


def test_the_stand_screen_says_where_its_calibration_goes_when_the_plate_was_registered_with_another_objective(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    assert view.calibrate_note.object == _routed(TEN, FORTY, 1)
    assert view.calibrate_note.visible is True
    assert view._stand_calibrate_for == FORTY
    # It follows the objective in place, at the next tick, and names that position's file.
    _turn(fake_microscope, TWENTY)
    view.tick()
    assert view.calibrate_note.object == _routed(TEN, TWENTY, 4)
    assert view._stand_calibrate_for == TWENTY
    # The button is offered and enabled: it keeps the plate's calibration, it does not warn.
    assert view.calibrate_button.disabled is False
    assert "goes out of date" not in view.calibrate_note.object


def test_the_stand_note_says_what_a_new_calibration_replaces_for_the_objectives_own_document_only(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # FM-172: what a new calibration replaces is said before the click; FM-247: one decision of
    # `again` for the block's button and this note (a usable document of THAT objective).
    held = _hold(run_doc, 1, FORTY)
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    assert view.calibrate_note.object == "\n".join(
        [_routed(TEN, FORTY, 1), calibrate_replaces(FORTY, SEEDED)]
    )
    assert (
        view.offsets_calibrate_button.label == f"Calibrate the camera again at {FORTY}"
    )

    # Another objective's calibration in the 40X's file is not the 40X's: nothing of its own to
    # replace, and the block's button agrees (no `again`).
    held.unlink()
    _hold(run_doc, 1, TWENTY)
    view.refresh()
    view.tick()
    assert view.calibrate_note.object == _routed(TEN, FORTY, 1)
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"

    # A document that is no calibration (it failed): nothing usable to replace.
    (run_doc.parent / "calibrations" / "position-1.json").write_text(
        "{not json", encoding="utf-8"
    )
    view.refresh()
    view.tick()
    assert view.calibrate_note.object == _routed(TEN, FORTY, 1)


def test_the_stand_screen_says_nothing_for_the_plates_own_objective_a_plate_by_eye_or_an_unknown_objective(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, TEN)
    # The plate's own objective: the calibration is the one the plate was registered with.
    assert view.calibrate_note.object == ""
    assert view.calibrate_note.visible is False

    # An objective not known: nothing to compare with.
    labels = fake_microscope.require(ObjectiveTurret).labels()
    fake_microscope.override(ObjectiveTurret, Turret(labels, None))
    view.tick()
    assert view.calibrate_note.object == ""

    # A plate registered by eye has no block, and the note says nothing at any objective.
    _seed_plate(run_doc, by_eye=True)
    fake_microscope.override(ObjectiveTurret, Turret(labels, FORTY))
    view.refresh()
    view.tick()
    assert view.offsets_box.visible is False
    assert view.calibrate_note.object == ""


def test_the_stand_note_follows_the_reason_and_the_busy_note_and_never_replaces_them(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()

    # After the busy note (`calibrate_wait`'s first reason here), never instead of it (D19).
    assert view.calibrate_note.object == "\n".join(
        [busy_note("slow", live=False), _routed(TEN, FORTY, 1)]
    )
    assert view.calibrate_button.disabled is True
    # The tool's own reason comes before them all.
    view._blocked = {**view._blocked, "camera-calibration": "the tool's own reason"}
    view.tick()
    assert view.calibrate_note.object == "\n".join(
        [
            "the tool's own reason",
            busy_note("slow", live=False),
            _routed(TEN, FORTY, 1),
        ]
    )
    Slow.release.set()
    assert view.page.join(10.0)


def test_the_stand_button_is_disabled_for_a_dry_run_a_light_that_reads_off_and_a_position_not_known(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
) -> None:
    # The note and the state are the controller's refusals, beside the button (FM-169): the same
    # `calibrate_wait`, and the position that the click would name.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    assert view.calibrate_button.disabled is False

    # The light reads off: its reason first, the destination after it, the button waits.
    light = Light(False)
    fake_microscope.override(LightSource, light)
    view.tick()
    assert view.calibrate_button.disabled is True
    assert view.calibrate_note.object == "\n".join(
        [CALIBRATE_LIGHT_OFF, _routed(TEN, FORTY, 1)]
    )
    # An unknown light refuses nothing; on neither.
    for reading in (None, True):
        light.reading = reading
        view.tick()
        assert view.calibrate_button.disabled is False

    # A position the page's copy of the labels does not name (the 40X on two positions): the
    # click would refuse it, and the note says so alone: it names a file it cannot.
    labels = fake_microscope.require(ObjectiveTurret).labels()
    labels[4] = FORTY
    fake_microscope.override(ObjectiveTurret, Turret(labels, FORTY))
    twice = make_view()
    assert twice.calibrate_button.disabled is True
    assert twice.calibrate_note.object == calibrate_position_unknown(FORTY)

    # A dry-run page that holds a real registration: the tool's own refusal, said first. Last:
    # closing it closes the core the fake stand shares.
    dry = Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )
    try:
        dry_view = make_view(dry)
        assert dry_view.calibrate_button.disabled is True
        assert dry_view.calibrate_note.object.splitlines()[0] == CALIBRATE_DRY_RUN
    finally:
        dry.close()


def test_the_stand_button_sends_the_label_it_showed_and_not_the_blocks(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-151, FM-177: at the plate's own objective the block's button names none (`_calibrate_for`
    # is `None`) and the Stand screen's names that objective: the click sends its own label and
    # the pairs the tool's own *Run* sends.
    view = _ready(make_view, fake_microscope, run_doc, TEN)
    assert view._calibrate_for is None
    assert view._stand_calibrate_for == TEN
    sent: list[tuple[str, tuple[str, ...]]] = []

    def spy(expected: str, pairs: Any = ()) -> None:
        sent.append((expected, tuple(pairs)))

    monkeypatch.setattr(view.page, "calibrate_camera", spy)
    view.tool_fields["camera-calibration"]["step_um"].value = "40"
    view.tool_fields["camera-calibration"]["from_run_doc"].value = "cal-10x.json"

    _click(view.calibrate_button)

    # Everything the form holds is sent: it is the controller that drops what a document does
    # not take (it chooses the destination, for the label it reads at the click).
    assert sent == [(TEN, ("step_um=40", "from_run_doc=cal-10x.json"))]
    monkeypatch.undo()

    # The label is the one drawn: a turn at the handset with no tick makes the click stale, and
    # the controller refuses it (the converse of FM-177: it would have chosen `run.json`).
    before = run_doc.read_bytes()
    _turn(fake_microscope, TWENTY)
    assert view._stand_calibrate_for == TEN
    _click(view.calibrate_button)
    assert view.page.status().outcome == (
        f"Camera calibration: {calibrate_changed(TEN, TWENTY)}"
    )
    assert StubCalibration.seen == []
    assert run_doc.read_bytes() == before
    assert not (run_doc.parent / "calibrations").exists()


def test_the_expert_views_run_camera_calibration_is_not_routed(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D19 changes the Stand screen's button only: the expert's *Run camera-calibration* is
    # `on_run`, the expert's way into `run.json` (which `plate-walls` reads), as it always was.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    routed: list[object] = []
    monkeypatch.setattr(view.page, "calibrate_camera", lambda *a, **k: routed.append(a))

    _click(view.tool_runs["camera-calibration"])
    assert view.page.join(10.0)

    assert routed == []
    record = RunDocument.load(run_doc).provider("camera-calibration")
    assert record is not None
    assert record.status is StageStatus.done
    assert not (run_doc.parent / "calibrations").exists()


def test_the_stands_command_is_the_routed_call_with_its_folder_note_and_the_run_documents_otherwise(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # ADR-0006, principle 5: the *Command* shows the call the click makes.
    view = _ready(make_view, fake_microscope, run_doc, FORTY)
    fields = view.tool_fields["camera-calibration"]
    fields["step_um"].value = "40"
    fields["from_run_doc"].value = "cal-10x.json"
    folder = run_doc.parent / "calibrations"

    def words() -> list[str]:
        return shlex.split(view.calibrate_command.object)

    # The routed call: the position's file, the step alone (a from_run_doc is refused by the
    # click and never sent), and the folder note under it.
    command = words()
    assert command[:3] == ["smc", "run", "camera-calibration"]
    assert command[command.index("--run-doc") + 1] == str(
        (folder / "position-1.json").absolute()
    )
    assert command[command.index("--param") + 1] == "step_um=40"
    assert "from_run_doc=cal-10x.json" not in command
    assert view.calibrate_command_note.object == (
        f"The folder {folder.absolute()} must exist first: the page makes it, a "
        "terminal does not."
    )

    # The plate's own objective: the run document's call as it always was, every field typed,
    # and no folder note.
    _turn(fake_microscope, TEN)
    view.tick()
    command = words()
    assert command[command.index("--run-doc") + 1] == str(run_doc.absolute())
    assert "from_run_doc=cal-10x.json" in command
    assert view.calibrate_command_note.object == ""

    # A position the page cannot name: no command to copy, for the click is refused.
    labels = fake_microscope.require(ObjectiveTurret).labels()
    labels[4] = FORTY
    fake_microscope.override(ObjectiveTurret, Turret(labels, FORTY))
    twice = make_view()
    assert twice.calibrate_command.object == ""
    assert twice.calibrate_command_note.object == ""
