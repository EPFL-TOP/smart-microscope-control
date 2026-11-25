"""The Wells screen's note on the camera calibration of the plan, and the command under *Lay out the fields*, without a browser (m5 §18.6 D26, #617).

The widgets are driven the way a session drives them: a click is ``button.clicks += 1`` and a typed
field is its ``.value``. The tool is a stub ``scan-plan`` (``Params = ScanPlanParams``) that keeps
what it was given and ends in a refusal; the controller's own tests
(``test_plate_page_plan_calibration.py``) hold the decision and the click, and
``tests/test_plate_page_plan_calibration_demo.py`` the whole chain on the demo stand with the real
tool. Every expected text is the design's, written out whole (FM-45). What this cannot show is a
browser: whether the note pushes *Lay out the fields* off a 620 px page, and whether a newcomer reads
it, are for ``/ui-check`` and the owner's walk (FM-161).
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.plugins.api import RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.plugin import ScanPlanParams
from smc.plugins.slots import (
    CameraCalibration,
    PlateResult,
    ScanPlanResult,
    WallImages,
    WellFit,
)
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The fake turret's positions, read from its definition and not from the code under test.
POSITION = {FORTY: 1, TEN: 2, TWENTY: 4}
FRAME = (384, 512)
POSIX_NOTE = (
    "that line is quoted for a POSIX shell (bash, zsh); PowerShell and cmd.exe "
    "read quotes differently: press the button, which makes the same call"
)


def _used(objective: str, position: int) -> str:
    """The note of a usable document, as m5 §18.6.3 words it (the date is the seeded one)."""
    return (
        f"Lay out the fields uses the camera calibration of {objective} made on "
        f"2026-11-25 (calibrations/position-{position}.json): the camera's turn and "
        "mirror are measured, not assumed."
    )


def _missing(objective: str, why: str) -> str:
    """The note of a document that cannot be used, as m5 §18.6.3 words it."""
    return (
        f"No camera calibration of {objective} can be used ({why}): Lay out the fields "
        "assumes the profile's pixel size, rotation 0 and no mirror, and the positions "
        "of cells in the fields will be wrong if the camera is turned or mirrored. To "
        f"get one: Stand screen, Calibrate the camera, with {objective} in place."
    )


# --- a stub scan-plan -------------------------------------------------------------------


class StubScanPlan:
    """``scan-plan`` that keeps what it was given and ends the run with a refusal."""

    name = provides = "scan-plan"
    summary = "A test tool."
    Params = ScanPlanParams
    Result = ScanPlanResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ("camera-calibration", "timings")
    seen: ClassVar[list[ScanPlanParams]] = []

    def run(
        self, microscope: Any, params: ScanPlanParams, ctx: RunContext
    ) -> ScanPlanResult:
        StubScanPlan.seen.append(params)
        raise PluginRunError("scan-plan", "the stub plans nothing")


@pytest.fixture(autouse=True)
def _fresh_state(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    StubScanPlan.seen = []
    yield
    # A raise in a screen's draw is logged and the tick goes on (``_guarded``): never expected here.
    assert [
        record.getMessage()
        for record in caplog.records
        if "the screen keeps what it drew last" in record.getMessage()
    ] == []


def _entry_points() -> list[EntryPoint]:
    return [EntryPoint(name="scan-plan", value=f"{__name__}:StubScanPlan", group=GROUP)]


# --- records written out by hand ----------------------------------------------------------


def _write(
    path: Path,
    plugin: str,
    provides: str,
    result: dict[str, Any] | None,
    *,
    params: dict[str, Any] | None = None,
    status: StageStatus = StageStatus.done,
    error: str | None = None,
) -> None:
    write_stage(
        path,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt="a0",
            status=status,
            started_utc=T0,
            finished_utc=T0,
            profile="demo",
            dry_run=False,
            smc_version="0",
            params=params if params is not None else {},
            result=result,
            error=error,
        ),
    )


def _seed_plate(run_doc: Path, *, by_eye: bool = False) -> None:
    """A plate registered under the 10X from its walls, or by eye (no offer)."""
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
    walls = WallImages(
        objective=TEN,
        pixel_size_um=1.1,
        image_shape_px=(100, 100),
        search_um=1000.0,
        wells=(fit,),
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
    )


def _hold(
    run_doc: Path,
    label: str,
    *,
    position: int | None = None,
    status: StageStatus = StageStatus.done,
    error: str | None = None,
) -> Path:
    """The document a calibration at ``label`` leaves in its turret position's file."""
    where = POSITION[label] if position is None else position
    path = run_doc.parent / "calibrations" / f"position-{where}.json"
    path.parent.mkdir(exist_ok=True)
    calibration = CameraCalibration(
        objective=label,
        image_shape_px=FRAME,
        matrix_um_per_px=((-0.2164, 0.1248), (-0.1246, -0.2164)),
        pixel_size_um=0.25,
        rotation_deg=29.9,
        mirrored=True,
        residual_um=0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=48.0,
        assumed=(),
    )
    _write(
        path,
        "camera-calibration",
        "camera-calibration",
        calibration.model_dump(mode="json") if status is StageStatus.done else None,
        status=status,
        error=error,
    )
    return path


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def stand(fake_core: FakeCore, fake_microscope: Microscope) -> Microscope:
    """The demo-like stand whose camera gives the frame the seeded documents were made in."""
    fake_core.image_shape = FRAME
    return fake_microscope


@pytest.fixture
def make_view(
    stand: Microscope, run_doc: Path
) -> Iterator[Callable[..., PlatePageView]]:
    """A view over the stand with the stub tool, wells A1 selected, drawn; every page it built is closed."""
    pages: list[PlatePage] = []

    def make(*, closed: bool = False) -> PlatePageView:
        page = (
            PlatePage.closed(stand.profile, run_doc, entry_points=_entry_points())
            if closed
            else PlatePage(stand, run_doc, entry_points=_entry_points())
        )
        pages.append(page)
        page.select_text("A1")
        view = PlatePageView(page)
        view._windows = False
        view.show_step("wells")
        view.refresh()
        view.tick()
        return view

    try:
        yield make
    finally:
        for page in pages:
            page.close()


def _turn(microscope: Microscope, label: str) -> None:
    """The operator's turn of the turret: what the page does not do, and these tests do not test."""
    microscope.require(ObjectiveTurret).select(label, confirm=True)


def _words(view: PlatePageView) -> list[str]:
    return shlex.split(view.layout_command.object)


def _calibration_pairs(view: PlatePageView) -> list[str]:
    command = _words(view)
    return [
        command[index + 1][len("calibration=") :]
        for index, word in enumerate(command[:-1])
        if word == "--param" and command[index + 1].startswith("calibration=")
    ]


def _same(found: str | Path | None, expected: Path) -> bool:
    return found is not None and os.path.samefile(found, expected)


# --- the note before the click --------------------------------------------------------------


def test_the_wells_screen_says_before_the_click_that_the_plan_assumes_rotation_0_and_no_mirror_and_where_to_get_the_document(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    _turn(stand, FORTY)
    view = make_view()

    # No calibration of the 40X yet: the page says what the plan will assume, and how to get one.
    assert view.layout_calibration_text.object == _missing(FORTY, "none has been made")
    assert view.layout_calibration_text.visible is True
    # The command is the call as it is: nothing is added for a document that is not there.
    assert _calibration_pairs(view) == []
    assert shlex.split(view.layout_command.object)[:3] == ["smc", "run", "scan-plan"]


def test_the_wells_screen_says_which_document_the_plan_uses_and_the_command_names_it(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    document = _hold(run_doc, FORTY)
    _turn(stand, FORTY)
    view = make_view()

    assert view.layout_calibration_text.object == _used(FORTY, 1)
    assert view.layout_calibration_text.visible is True
    # The command is the call the click makes, with the document once, absolute.
    (given,) = _calibration_pairs(view)
    assert Path(given).is_absolute()
    assert _same(given, document)
    # And the click gives the tool that same document (the note and the command are the tick's
    # copy of the decision the click reads again).
    view.layout_button.clicks += 1
    assert view.page.join(10.0)
    (seen,) = StubScanPlan.seen
    assert _same(seen.calibration, document)
    assert os.path.samefile(seen.calibration or "", given)


def test_the_note_names_each_reason_a_document_cannot_be_used(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    _hold(run_doc, FORTY, status=StageStatus.failed, error="the frame is dark")
    _turn(stand, FORTY)
    view = make_view()
    # A failed record says the tool's own words, in the held list's form.
    assert view.layout_calibration_text.object == _missing(
        FORTY, "the last calibration failed: the frame is dark"
    )
    assert _calibration_pairs(view) == []

    # The 20X's position holds the 10X's calibration (a turn during a run, a relabelled turret).
    _hold(run_doc, TEN, position=POSITION[TWENTY])
    _turn(stand, TWENTY)
    view.refresh()
    view.tick()
    assert view.layout_calibration_text.object == _missing(
        TWENTY, f"calibrations/position-4.json holds a calibration of {TEN}"
    )


@pytest.mark.parametrize(
    "case",
    ["the plate's own objective", "no offer", "the stand closed"],
)
def test_the_note_is_empty_and_takes_no_height_at_the_plates_own_objective_with_no_offer_or_with_the_stand_closed(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
    case: str,
) -> None:
    _seed_plate(run_doc, by_eye=case == "no offer")
    _hold(run_doc, FORTY)
    _turn(stand, TEN if case == "the plate's own objective" else FORTY)
    view = make_view(closed=case == "the stand closed")
    if case == "the stand closed":
        # An objective typed in the form is an objective all the same: with the stand closed the
        # note says nothing about a stand that is not there (FM-145), and does not wait for a
        # turret that cannot be read to say `the turret position ... is not known`.
        view.tool_fields["scan-plan"]["objective"].value = FORTY
        view.tick()

    assert view.layout_calibration_text.object == ""
    assert view.layout_calibration_text.visible is False
    assert _calibration_pairs(view) == []


def test_the_note_is_not_a_wait_the_lay_out_button_stays_enabled(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    # FM-169's converse: the operator may want rotation 0, and the click is the controller's.
    _seed_plate(run_doc)
    _turn(stand, FORTY)
    view = make_view()

    assert view.layout_calibration_text.object == _missing(FORTY, "none has been made")
    assert view.layout_button.disabled is False
    assert view.layout_wait.object == ""
    view.layout_button.clicks += 1
    assert view.page.join(10.0)
    (seen,) = StubScanPlan.seen
    assert seen.calibration is None


def test_a_turn_at_the_handset_changes_the_note_and_the_command_at_the_next_tick(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    forty = _hold(run_doc, FORTY)
    twenty = _hold(run_doc, TWENTY)
    _turn(stand, FORTY)
    view = make_view()
    assert view.layout_calibration_text.object == _used(FORTY, 1)
    assert _same(_calibration_pairs(view)[0], forty)

    # No refresh between: the tick alone follows the operator's turn.
    _turn(stand, TWENTY)
    view.tick()
    assert view.layout_calibration_text.object == _used(TWENTY, 4)
    (given,) = _calibration_pairs(view)
    assert _same(given, twenty)

    # The plate's own objective: nothing to say, and the pair is gone from the command.
    _turn(stand, TEN)
    view.tick()
    assert view.layout_calibration_text.object == ""
    assert view.layout_calibration_text.visible is False
    assert _calibration_pairs(view) == []
    # The command is still the call: the wells the page filled stay in it.
    assert "wells=A1" in _words(view)


def test_a_typed_calibration_or_pixel_size_silences_the_note_and_the_command_holds_only_the_typed_one(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
    tmp_path: Path,
) -> None:
    _seed_plate(run_doc)
    document = _hold(run_doc, FORTY)
    _turn(stand, FORTY)
    view = make_view()
    fields = view.tool_fields["scan-plan"]
    assert view.layout_calibration_text.object == _used(FORTY, 1)

    # A typed calibration is the only one: no note, and the command never holds two.
    own = tmp_path / "mine.json"
    fields["calibration"].value = str(own)
    assert view.layout_calibration_text.object == ""
    assert view.layout_calibration_text.visible is False
    assert _calibration_pairs(view) == [str(own)]

    # A typed pixel size: no note and no pair (the tool refuses both, `not both`).
    fields["calibration"].value = ""
    fields["pixel_size_um"].value = "0.3"
    assert view.layout_calibration_text.object == ""
    assert _calibration_pairs(view) == []
    assert "pixel_size_um=0.3" in _words(view)

    # `null` is unset, as the parser reads it: the page's pair replaces a typed `calibration=null`.
    fields["pixel_size_um"].value = ""
    fields["calibration"].value = "null"
    assert view.layout_calibration_text.object == _used(FORTY, 1)
    (given,) = _calibration_pairs(view)
    assert _same(given, document)
    assert "calibration=null" not in _words(view)

    # Cleared again: the note and the pair are back.
    fields["calibration"].value = ""
    assert view.layout_calibration_text.object == _used(FORTY, 1)
    assert len(_calibration_pairs(view)) == 1


def test_a_typed_objective_is_the_plans_objective_for_the_note_and_the_command(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    _hold(run_doc, FORTY)
    twenty = _hold(run_doc, TWENTY)
    _turn(stand, FORTY)
    view = make_view()
    objective = view.tool_fields["scan-plan"]["objective"]

    objective.value = TWENTY
    assert view.layout_calibration_text.object == _used(TWENTY, 4)
    assert _same(_calibration_pairs(view)[0], twenty)
    assert f"objective={TWENTY}" in _words(view)

    # The plate's own objective typed: its calibration is the run document's, nothing to say.
    objective.value = TEN
    assert view.layout_calibration_text.object == ""
    assert _calibration_pairs(view) == []

    # `null` is the objective in place.
    objective.value = "null"
    assert view.layout_calibration_text.object == _used(FORTY, 1)


@pytest.mark.parametrize("windows", [False, True], ids=["posix page", "windows page"])
def test_the_command_with_a_calibration_path_keeps_the_quoting_note_on_a_windows_page(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
    windows: bool,
) -> None:
    # FM-175: the line is quoted for a POSIX shell; a path in it does not change that, and the
    # note stays under it. The platform is set both ways, never left to the runner's own.
    _seed_plate(run_doc)
    _hold(run_doc, FORTY)
    _turn(stand, FORTY)
    view = make_view()
    view._windows = windows
    view.refresh()
    view.tick()

    assert len(_calibration_pairs(view)) == 1
    assert view.layout_command_note.object == (POSIX_NOTE if windows else "")


# --- what the tick reads ------------------------------------------------------------------


def test_the_note_reads_no_stand_and_no_file_across_ticks(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-145: the tick draws the note and, when the document it names changes, the command from
    # the last redraw's pairs: no document is read, no stand is read for it, no `page_params`
    # (which reads the run document) and no `view()`.
    _seed_plate(run_doc)
    forty = _hold(run_doc, FORTY)
    twenty = _hold(run_doc, TWENTY)
    _turn(stand, FORTY)
    view = make_view()
    calls: list[str] = []
    # The turret's labels are the page's cached copy (`_objective_labels`), never read again
    # through `objective_labels` or `_labels_of`, each of which is a stand read.
    for name in (
        "_held_calibrations",
        "_plan_reading",
        "objective_labels",
        "_labels_of",
        "view",
        "page_params",
    ):
        real = getattr(PlatePage, name)

        def spy(
            self: PlatePage, *args: Any, _real: Any = real, _name: str = name, **kw: Any
        ) -> Any:
            # `page_params` of the other tools is the tick's own (no run document is read for
            # them); the one that reads the run document is `scan-plan`'s.
            if _name != "page_params" or args[:1] == ("scan-plan",):
                calls.append(_name)
            return _real(self, *args, **kw)

        monkeypatch.setattr(PlatePage, name, spy)

    drawn: list[str] = []
    for label in (FORTY, FORTY, TWENTY, TWENTY, TEN, TEN, FORTY, TWENTY, FORTY, FORTY):
        _turn(stand, label)
        view.tick()
        drawn.append(view.layout_calibration_text.object)

    # The path changed several times (and so was the command drawn again), with no read.
    assert drawn[0] == _used(FORTY, 1)
    assert drawn[2] == _used(TWENTY, 4)
    assert drawn[4] == ""
    assert calls == []
    assert _same(_calibration_pairs(view)[0], forty)
    view_pairs = _calibration_pairs(view)
    assert len(view_pairs) == 1
    assert not _same(view_pairs[0], twenty)


def test_the_note_is_drawn_again_when_a_document_is_made_and_the_view_is_read_again(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    # *Calibrate the camera* on the Stand screen ends a run: the page reads the view again and the
    # Wells screen's note follows it (the document the run made is now held).
    _seed_plate(run_doc)
    _turn(stand, FORTY)
    view = make_view()
    assert view.layout_calibration_text.object == _missing(FORTY, "none has been made")

    document = _hold(run_doc, FORTY)
    view.refresh()

    assert view.layout_calibration_text.object == _used(FORTY, 1)
    (given,) = _calibration_pairs(view)
    assert _same(given, document)


def test_the_command_has_no_pair_when_the_controller_refuses_the_wells(
    make_view: Callable[..., PlatePageView],
    stand: Microscope,
    run_doc: Path,
) -> None:
    # No wells selected: no command to copy (the click is refused), and a tick keeps it so even
    # when the document the note names changes.
    _seed_plate(run_doc)
    _hold(run_doc, FORTY)
    _hold(run_doc, TWENTY)
    _turn(stand, FORTY)
    view = make_view()
    view.page.select_text("")
    view.refresh()
    assert view.layout_command.object == ""
    assert view.layout_card.visible is False

    _turn(stand, TWENTY)
    view.tick()

    assert view.layout_command.object == ""
    assert view.layout_card.visible is False
    # The note is about the plan, not the selection: it follows the turn all the same.
    assert view.layout_calibration_text.object == _used(TWENTY, 4)
