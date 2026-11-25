"""The Stand screen's pane with the numbers of a calibration, without a browser (m5 §23.2 D1 to D5, #627).

The widgets are driven the way a session drives them: a tick after a refresh. The documents are
seeded by hand with the numbers the demo measured (m5 §23.1), so that every expected text is the
design's, written out whole and never built by the code under test (FM-45, FM-203). The real tool
and the click that writes the document are ``tests/test_plate_page_calibration_card_demo.py``'s.
What this cannot show is a browser: whether the pane pushes the Stand screen's first action off a
620 px page, and what a browser makes of a ``Str`` pane's angle brackets (FM-271), are for the
one-off session of ``/ui-check`` and the owner's walk (FM-161).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.errors import HardwareError
from smc.plugins.registry import GROUP
from smc.plugins.slots import CameraCalibration, PlateResult, WallImages, WellFit
from smc.run import StageRecord, StageStatus, write_stage
from smc.ui.controller import PlatePage
from smc.ui.guide import local_time
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

T0 = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
TEN = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"
#: The fake turret's positions, read from its definition and not from the code under test.
POSITION = {FORTY: 1, TEN: 2}
#: Markup in a label, a path and a note, as the stand and a document could hold it (FM-139).
HOSTILE = "<img src=x onerror=alert(1)> & <b>40X</b>"
LOCAL = local_time(T0)
RUN_DOC_SAYS = "the run document's own line"
HELD_SAYS = "the position's own line"

#: The demo's numbers (m5 §23.1), and what the pane says for them (D3).
DEMO_LINE = "Pixel size 0.2500 µm, rotation 30.0°, mirrored, residual 0.04 µm, probe step 32 µm."
#: The run document's calibration (the 10X's): numbers of its own, distinct from the demo's.
RUN_LINE = (
    "Pixel size 0.5000 µm, rotation 0.0°, not mirrored, residual 0.01 µm, "
    "probe step 48 µm."
)


def _calibration(
    objective: str | None,
    *,
    demo: bool,
    assumed: tuple[str, ...] = (),
) -> CameraCalibration:
    """The calibration of the 40X on the demo (``demo=True``) or one with numbers of its own."""
    return CameraCalibration(
        objective=objective,
        image_shape_px=(384, 512),
        matrix_um_per_px=((0.2166, -0.1249), (-0.1249, -0.2166))
        if demo
        else ((0.5, 0.0), (0.0, -0.5)),
        pixel_size_um=0.249966 if demo else 0.5,
        rotation_deg=29.9891 if demo else 0.0,
        mirrored=demo,
        residual_um=0.0437417 if demo else 0.01,
        response=0.97,
        return_error_um=0.4,
        step_um=32.0 if demo else 48.0,
        assumed=assumed,
    )


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


def _seed_run_calibration(
    run_doc: Path, objective: str | None = TEN, *, assumed: tuple[str, ...] = ()
) -> None:
    """The run document's own ``camera-calibration``, with the numbers of ``RUN_LINE``."""
    _write(
        run_doc,
        "camera-calibration",
        "camera-calibration",
        _calibration(objective, demo=False, assumed=assumed).model_dump(mode="json"),
    )


def _seed_plate(run_doc: Path, objective: str = TEN) -> None:
    """A plate registered from its walls under ``objective``: the offer (m5 §18.2 D2)."""
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
        objective=objective,
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
        method="wall-images",
        residual_um=1.0,
        scale_error=None,
        n_refs=2,
        walls=walls,
        objective=objective,
    )
    _write(
        run_doc,
        "plate-walls",
        "plate",
        plate.model_dump(mode="json"),
        params={"plate": plate.plate},
    )


def _hold(
    run_doc: Path,
    label: str,
    *,
    assumed: tuple[str, ...] = (),
    objective: str | None = None,
) -> Path:
    """The document a calibration at ``label`` leaves in its position's file, with the demo's numbers."""
    path = run_doc.parent / "calibrations" / f"position-{POSITION[label]}.json"
    path.parent.mkdir(exist_ok=True)
    _write(
        path,
        "camera-calibration",
        "camera-calibration",
        _calibration(objective or label, demo=True, assumed=assumed).model_dump(
            mode="json"
        ),
    )
    return path


def _entry_points() -> list[EntryPoint]:
    return [
        EntryPoint(
            name="camera-calibration",
            value="smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin",
            group=GROUP,
        )
    ]


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def make_view(
    fake_microscope: Microscope, run_doc: Path
) -> Iterator[Callable[..., PlatePageView]]:
    """A view on the Stand screen over the fake stand (or a closed one), drawn; every page it built is closed."""
    pages: list[PlatePage] = []

    def make(*, closed: bool = False) -> PlatePageView:
        page = (
            PlatePage.closed(
                fake_microscope.profile, run_doc, entry_points=_entry_points()
            )
            if closed
            else PlatePage(fake_microscope, run_doc, entry_points=_entry_points())
        )
        pages.append(page)
        view = PlatePageView(page)
        view.show_step("stand")
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


def _redraw(view: PlatePageView) -> None:
    view.refresh()
    view.tick()


class Turret:
    """A turret the test sets: the label in place and the positions' labels."""

    def __init__(self, labels: list[str], now: str | None) -> None:
        self.labels_ = labels
        self.now = now
        self.reads: list[str] = []

    def labels(self) -> list[str]:
        self.reads.append("labels")
        return list(self.labels_)

    def current(self) -> str:
        self.reads.append("current")
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


# --- where the pane is and what it says ---------------------------------------------------


def test_the_stand_screen_says_the_numbers_of_the_run_documents_calibration_under_the_known_block(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    _seed_run_calibration(run_doc, TEN, assumed=(RUN_DOC_SAYS,))

    view = make_view()

    assert view.calibration_numbers.object == "\n".join(
        [
            f"Camera calibration at {TEN}, measured {LOCAL}.",
            RUN_LINE,
            f"Written to {run_doc}.",
            f"assumed: {RUN_DOC_SAYS}",
        ]
    )
    assert view.calibration_numbers.visible
    # Directly under the known block, which says what it said, and outside the box that hides.
    screen = view.screens["stand"]
    at = screen.index(view.known_block)
    assert screen[at + 1] is view.calibration_numbers
    assert screen[at + 2] is view.calibrate_box
    assert view.calibration_numbers not in view.calibrate_box.objects
    assert view.known_block.object.split("\n")[1] == (
        f"Camera: calibrated at {TEN} on {LOCAL}"
    )
    # The pane is on the Stand screen only: no other screen lists it.
    assert [
        name
        for name, objects in view.screens.items()
        if view.calibration_numbers in objects
    ] == ["stand"]


def test_a_closed_stand_still_says_the_run_documents_numbers(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    # D1 item 1: the objective is not known, so the run document's calibration is the pane's, as
    # the known block's `Camera:` line speaks of it.
    _seed_run_calibration(run_doc, FORTY)

    view = make_view(closed=True)

    assert view.calibration_numbers.object.split("\n")[0] == (
        f"Camera calibration at {FORTY}, measured {LOCAL}."
    )
    assert view.calibration_numbers.object.split("\n")[1] == RUN_LINE


def test_at_another_objective_the_stand_screen_says_the_numbers_of_its_document_and_leaves_the_known_block_as_it_was(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    # The plate was registered under the 10X, the 40X is in place (§18.6.1): the click writes into
    # position 1's document and the known block still speaks of the run document's (D19).
    _seed_plate(run_doc)
    _seed_run_calibration(run_doc, TEN)
    _turn(fake_microscope, FORTY)
    view = make_view()
    known = f"Camera: calibrated at {TEN} on {LOCAL}, not at {FORTY}"

    # Nothing is held for the 40X: the 10X's numbers are not shown as the 40X's (FM-216).
    assert view.calibration_numbers.object == ""
    assert not view.calibration_numbers.visible
    assert view.known_block.object.split("\n")[1] == known

    path = _hold(run_doc, FORTY, assumed=(HELD_SAYS,))
    _redraw(view)

    assert view.calibration_numbers.object == "\n".join(
        [
            f"Camera calibration at {FORTY}, measured {LOCAL}.",
            DEMO_LINE,
            f"Written to {path}.",
            f"assumed: {HELD_SAYS}",
        ]
    )
    assert view.calibration_numbers.visible
    assert view.known_block.object.split("\n")[1] == known

    # Back at the plate's own objective: the run document's calibration, with its own path.
    _turn(fake_microscope, TEN)
    _redraw(view)
    assert view.calibration_numbers.object.split("\n")[1] == RUN_LINE
    assert view.calibration_numbers.object.split("\n")[2] == f"Written to {run_doc}."


def test_a_document_that_is_not_its_positions_says_nothing_on_the_screen(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    _seed_plate(run_doc)
    _turn(fake_microscope, FORTY)
    # The 40X's file holds the 10X's calibration (the turret was turned during the run).
    _hold(run_doc, FORTY, objective=TEN)
    view = make_view()

    assert view.calibration_numbers.object == ""
    assert not view.calibration_numbers.visible


def test_the_pane_stays_when_the_calibrate_box_hides_and_takes_no_height_while_empty(
    make_view: Callable[..., PlatePageView], run_doc: Path
) -> None:
    view = make_view()
    # Not calibrated: the button is offered and the pane has nothing to say, so it takes no height.
    assert view.calibrate_box.visible
    assert view.calibration_numbers.object == ""
    assert view.calibration_numbers.visible is False

    _seed_run_calibration(run_doc, TEN)
    _redraw(view)

    # Calibrated at the objective in place: the box is gone and the numbers are on the screen.
    assert not view.calibrate_box.visible
    assert view.calibration_numbers.visible
    assert view.calibration_numbers.object.split("\n")[1] == RUN_LINE


# --- plain text, and what the tick reads (D5, FM-139, FM-145) -------------------------------


def test_the_pane_is_plain_text_even_for_a_label_that_looks_like_markup(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
) -> None:
    """FM-139: a label, a note and a path come from outside; the pane says them as given.

    This pins the model's text and its type (a ``Str``, not Markdown). That a browser shows the
    angle brackets as text is **not** proved by the pane's type (FM-271: it parses a ``Str``
    pane as HTML); that is the one-off session's look.
    """
    labels = [HOSTILE, TEN]
    fake_microscope.override(ObjectiveTurret, Turret(labels, HOSTILE))
    _seed_run_calibration(run_doc, HOSTILE, assumed=(f"a note: {HOSTILE}",))

    view = make_view()

    assert type(view.calibration_numbers) is pn.pane.Str
    assert view.calibration_numbers.object.split("\n") == [
        f"Camera calibration at {HOSTILE}, measured {LOCAL}.",
        RUN_LINE,
        f"Written to {run_doc}.",
        f"assumed: a note: {HOSTILE}",
    ]


def test_drawing_the_pane_reads_no_device(
    make_view: Callable[..., PlatePageView],
    fake_microscope: Microscope,
    run_doc: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FM-145: the pane is drawn from the status and the last view; the draw adds no read of the
    # turret to the one the status makes at a tick (a drawer that read `labels()` would take the
    # core's lock at every tick of every tab), and reads no document.
    _seed_plate(run_doc)
    _seed_run_calibration(run_doc, TEN)
    _hold(run_doc, FORTY)
    turret = Turret(list(fake_microscope.require(ObjectiveTurret).labels()), FORTY)
    view = make_view()
    fake_microscope.override(ObjectiveTurret, turret)
    view.tick()
    baseline = list(turret.reads)
    page_reads: list[str] = []
    for name in ("view", "_held_calibrations", "objective_labels", "_turret_label"):
        real = getattr(PlatePage, name)

        def spy(
            self: PlatePage, *args: Any, _real: Any = real, _name: str = name, **kw: Any
        ) -> Any:
            page_reads.append(_name)
            return _real(self, *args, **kw)

        monkeypatch.setattr(PlatePage, name, spy)

    for label in (TEN, FORTY, TEN):
        turret.now = label
        turret.reads.clear()
        view.tick()
        assert turret.reads == baseline, turret.reads
    assert "labels" not in baseline
    assert page_reads == []
    assert view.calibration_numbers.object.split("\n")[1] == RUN_LINE
