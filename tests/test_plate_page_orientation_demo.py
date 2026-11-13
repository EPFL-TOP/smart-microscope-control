"""The *Plate lies* switch on the demo stand, over the real runner and the real plate tools (m5 §16, #422).

A 24-well (4 x 6, not square: FM-73) lies turned 180° in its holder and tilted 0.3°, A1
at the stage's +X -Y corner. The page registers it from three centres, declared turned,
and the stage stands on the printed B3. Every expected position is ``useq``'s own
(``WellPlatePlan(rotation=180 + tilt)``), never ``smc``'s geometry: a mirror in place
of a turn, or rows swapped with columns, fails here.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, tzinfo
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
import useq

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.plugins.registry import GROUP
from smc.run import RunDocument, StageStatus
from smc.ui.controller import PlatePage
from smc.ui.plate_page import PlatePageView

pytestmark = [
    pytest.mark.demo,
    # A failure a handler swallowed fails the test that did not expect it (#277).
    pytest.mark.usefixtures("handler_failures"),
]

PLATE = "24-well"
#: A1 of the plate as it lies, turned: on the stage's +X -Y side of the plate.
A1_UM = (48000.0, -29000.0)
TILT_DEG = 0.3
TOOLS = {
    "goto-well": "smc.plugins.goto_well:GotoWellPlugin",
    "plate-orientation": "smc.plugins.plate_orientation:PlateOrientationPlugin",
    "plate-registration": "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
}


def _truth(well: str) -> tuple[float, float]:
    """Where the printed ``well`` is on the stage, from ``useq``'s plan of the turned plate."""
    plan = useq.WellPlatePlan(
        plate=PLATE, a1_center_xy=A1_UM, rotation=180.0 + TILT_DEG
    )
    (position,) = [p for p in plan.all_well_positions if p.name == well]
    return float(position.x), float(position.y)  # type: ignore[arg-type]


class _Ticking(datetime):
    """``datetime`` whose ``now`` never gives the same instant twice: 20 ms apart at least.

    The latest start provides a slot, and a tie goes to the later plugin name, which
    ``plate-registration`` is: a turn started in the clock tick of the fit it follows
    would hide behind it (FM-197, FM-219).
    """

    last: datetime | None = None

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _Ticking:
        real = datetime.now(tz)
        if cls.last is not None and real < cls.last + timedelta(milliseconds=20):
            real = cls.last + timedelta(milliseconds=20)
        cls.last = real
        return cls.fromtimestamp(real.timestamp(), tz)


@pytest.fixture(autouse=True)
def _starts_never_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    _Ticking.last = None
    monkeypatch.setattr("smc.plugins.runner.datetime", _Ticking)


@pytest.fixture
def registered(
    demo_microscope: Microscope, tmp_path: Path
) -> Iterator[tuple[PlatePage, PlatePageView]]:
    """The page with the turned 24-well registered from A1, A6 and D1, declared turned, the stage on B3."""
    demo_microscope.core.setProperty("XY", "Velocity", "100000")
    page = PlatePage(
        demo_microscope,
        tmp_path / "run.json",
        entry_points=[
            EntryPoint(name=name, value=value, group=GROUP)
            for name, value in TOOLS.items()
        ],
    )
    try:
        view = PlatePageView(page)
        view.plate_select.value = PLATE
        # Declared before the fit: an estimate's choice, which the fit is given.
        view.orientation_select.value = "turned-180"
        assert page.orientation == "turned-180"
        stage = demo_microscope.require(XYStage)
        for well in ("A1", "A6", "D1"):
            stage.move_to_um(*_truth(well))
            page.capture_centre(well)
        page.run("plate-registration", page.page_params("plate-registration"))
        assert page.join(30.0)
        record = RunDocument.load(tmp_path / "run.json").stages["plate-registration"]
        assert record.status is StageStatus.done, page.status().outcome
        assert record.params["orientation"] == "turned-180"
        stage.move_to_um(*_truth("B3"))
        view.refresh()
        view.tick()
        yield page, view
    finally:
        page.close()


def _cross(view: PlatePageView) -> dict[str, list[Any]]:
    return {key: list(value) for key, value in view.stage_source.data.items()}


def test_switching_renames_the_wells_and_the_cross_stays(
    registered: tuple[PlatePage, PlatePageView], demo_microscope: Microscope
) -> None:
    page, view = registered
    run_doc = page.run_doc
    stage = demo_microscope.require(XYStage)
    before = _cross(view)
    assert before["text"] == ["over B3"]
    where = stage.position_um()
    fit = RunDocument.load(run_doc).stages["plate-registration"]

    # The operator sees that the objective is over C4, not B3: the plate lies the
    # other way. One switch, and the names follow.
    view.orientation_select.value = "not-turned"
    assert page.join(10.0)
    view.tick()

    after = _cross(view)
    assert after["x"] == before["x"]
    assert after["y"] == before["y"]
    assert after["text"] == ["over C4"]
    assert view.orientation_select.value == "not-turned"
    # No stage move, and no new fit: the turn is a run of plate-orientation.
    moved = stage.position_um()
    assert (moved.x_um, moved.y_um) == (where.x_um, where.y_um)
    doc = RunDocument.load(run_doc)
    assert doc.stages["plate-registration"] == fit
    assert doc.stages["plate-orientation"].status is StageStatus.done
    assert doc.provider("plate") == doc.stages["plate-orientation"]


def test_selecting_b3_on_a_turned_plate_drives_to_the_printed_b3(
    registered: tuple[PlatePage, PlatePageView], demo_microscope: Microscope
) -> None:
    page, _view = registered
    stage = demo_microscope.require(XYStage)
    stage.move_to_um(0.0, 0.0)
    # A drive click on the map's B3: the map is the turned plate's, so its B3 is the
    # printed B3, and goto-well reads the same registered plate.
    b3 = next(w for w in page.view().wells if w.well == "B3")
    assert page.click(*b3.center_um, drive=True) == "B3"
    assert page.join(30.0)
    outcome = page.status().outcome
    assert outcome is not None
    assert outcome.startswith("Go to done"), outcome
    landed = stage.position_um()
    x_um, y_um = _truth("B3")
    assert abs(landed.x_um - x_um) < 1.0
    assert abs(landed.y_um - y_um) < 1.0
