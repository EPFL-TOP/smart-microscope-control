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
from smc.hardware.capabilities import ObjectiveTurret
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
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui import plate_page
from smc.ui.controller import PlatePage
from smc.ui.guide import (
    CALIBRATE_DRY_RUN,
    CALIBRATE_LEAD,
    CALIBRATE_NOT_KNOWN,
    NOT_CONNECTED,
    OFFSETS_NO_DOCUMENT,
    busy_note,
    calibrate_changed,
    calibrate_is_the_plates,
    calibrate_replaces,
    calibrate_replaces_plate,
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
    step_um: float | None = None
    settle_ms: float = 0.0


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
    # The expert form's own field, typed: what the Stand screen's button sends too.
    view.tool_fields["camera-calibration"]["step_um"].value = "40"

    _click(view.offsets_calibrate_button)

    assert sent == [(FORTY, ("step_um=40",))]


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
        "Nikon 60X Plan Apo: calibrated on 2026-11-17; not on the turret now: "
        "calibrate its position again, or type the documents to use",
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
    # A failed record leaves the position with no usable document: no `again`.
    assert view.offsets_calibrate_button.label == f"Calibrate the camera at {FORTY}"
    assert f"{FORTY}: no calibration yet" in lines


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
    for name in ("view", "_held_calibrations", "offsets_step", "_halted"):
        monkeypatch.setattr(view.page, name, lambda *a, _n=name: calls.append(_n) or ())
    for label in (TWENTY, FORTY, TEN):
        _turn(fake_microscope, label)
        view.tick()

    assert calls == []
    assert view.offsets_calibrate_button.label == ANOTHER


# --- the Stand screen's button says what it replaces (D17, FM-216) ---------------------


def test_the_stand_screen_says_what_its_calibration_replaces_when_the_plate_was_registered_with_another_objective(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    assert view.calibrate_note.object == calibrate_replaces_plate(TEN, FORTY)
    assert view.calibrate_note.visible is True
    # It follows the objective in place, at the next tick.
    _turn(fake_microscope, TWENTY)
    view.tick()
    assert view.calibrate_note.object == calibrate_replaces_plate(TEN, TWENTY)
    # The button itself still runs the tool into the run document: it is not routed nor hidden.
    assert view.calibrate_button.disabled is False


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


def test_the_stand_screen_note_follows_the_tools_reason_and_the_busy_note(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    view = _ready(make_view, fake_microscope, run_doc, FORTY)

    _click(view.tool_runs["slow"])
    assert Slow.started.wait(5)
    view.tick()

    # After the busy note, never instead of it (D17).
    assert view.calibrate_note.object == "\n".join(
        [busy_note("slow", live=False), calibrate_replaces_plate(TEN, FORTY)]
    )
    Slow.release.set()
    assert view.page.join(10.0)
