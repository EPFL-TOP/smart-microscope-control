"""Where the map draws the plate before and after a registration: its four poses (m5 §15 D3, #410).

``PlateView.pose`` says which of *registered*, *seed*, *estimated* and *unplaced* the map is, and
the wells are drawn where that pose puts them. These tests drive the controller alone (no Panel,
no browser), on the facade over ``FakeCore`` with a profile that does or does not carry a
``[holder]``. The 96-well plate's grid is not square (12 columns by 8 rows), so swapped axes or a
flipped sign show (FM-73); the Greiner 24-well is the lab's own plate (#400).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult, WellRef, holder_estimate, well_geometry
from smc.plugins.slots.plate import PlateOrientation
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.testing.sample_stand import sample_refs
from smc.ui.controller import MapPose, PlatePage, PlateView
from smc.ui.guide import (
    GOTO_UNPLACED,
    GOTO_WALL_NEEDS_REGISTRATION,
    PLATE_NOT_CHOSEN,
)

PROFILES = Path(__file__).resolve().parents[2] / "profiles"
HOLDER_CENTRE = (400.0, -325.0)
MEASURED = "2026-11-10, sessions 2 and 3 (#334, #405): the mean of two centres"
THEN = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
#: A1's corner on a stand that does not say its sides (m5 §16 D3): the demo's.
STAGE_A1 = "stage -X +Y"
GOTO = EntryPoint(
    name="goto-well", value="smc.plugins.goto_well:GotoWellPlugin", group=GROUP
)


def _with_holder(centre: tuple[float, float] | None = HOLDER_CENTRE) -> Profile:
    """The demo profile with a ``[holder]`` (or, with ``None``, as it is: no table)."""
    demo = Profile.demo()
    if centre is None:
        return demo
    return demo.model_copy(
        update={"holder": HolderSection(plate_centre_xy_um=centre, measured=MEASURED)}
    )


def _page(
    core: FakeCore, run_doc: Path, profile: Profile
) -> Iterator[tuple[PlatePage, Microscope]]:
    microscope = Microscope.from_core(cast("CMMCorePlus", core), profile)
    # The real goto-well: a drive click on the map runs it (m3 §5.5).
    page = PlatePage(microscope, run_doc, entry_points=[GOTO])
    try:
        yield page, microscope
    finally:
        page.close()
        microscope.close()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def holder_page(fake_core: FakeCore, run_doc: Path) -> Iterator[PlatePage]:
    for page, _ in _page(fake_core, run_doc, _with_holder()):
        yield page


@pytest.fixture
def bare_page(fake_core: FakeCore, run_doc: Path) -> Iterator[PlatePage]:
    for page, _ in _page(fake_core, run_doc, _with_holder(None)):
        yield page


@pytest.fixture
def holder_stand(
    fake_core: FakeCore, run_doc: Path
) -> Iterator[tuple[PlatePage, Microscope]]:
    """``holder_page`` and the facade it drives, to read where the stage went."""
    yield from _page(fake_core, run_doc, _with_holder())


@pytest.fixture
def bare_stand(
    fake_core: FakeCore, run_doc: Path
) -> Iterator[tuple[PlatePage, Microscope]]:
    """``bare_page`` and its facade."""
    yield from _page(fake_core, run_doc, _with_holder(None))


def _register(
    run_doc: Path,
    plate: PlateResult,
) -> None:
    """``plate`` as an earlier tool left it in the run document, ``done`` and of this stand."""
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


def _plate_result(
    plate: str = "96-well",
    a1: tuple[float, float] = (1000.0, -500.0),
    method: str = "references",
    refs: tuple[WellRef, ...] = (),
) -> PlateResult:
    seeded = method == "seed"
    return PlateResult(
        plate=plate,
        a1_center_xy_um=a1,
        rotation_deg=0.0 if seeded else 3.0,
        method=method,  # type: ignore[arg-type]
        residual_um=None if seeded else 2.0,
        scale_error=None if seeded else 0.001,
        n_refs=len(refs) if seeded else 2,
        refs=refs,
    )


def _centres(view: PlateView) -> dict[str, tuple[float, float]]:
    return {well.well: well.center_um for well in view.wells}


def test_with_no_plate_and_no_holder_the_map_is_unplaced_with_a1_at_the_origin(
    bare_page: PlatePage,
) -> None:
    view = bare_page.view()

    assert view.pose == MapPose("unplaced")
    assert view.nominal is True
    assert _centres(view)["A1"] == (0.0, 0.0)


@pytest.mark.parametrize(
    ("plate", "a1_um", "wells"),
    [
        ("96-well", (-49_100.0, 31_175.0), 96),
        ("greiner-24-well", (-48_225.0, 28_850.0), 24),
    ],
)
def test_a_holder_places_the_nominal_plate_by_its_grid_centre(
    holder_page: PlatePage,
    plate: str,
    a1_um: tuple[float, float],
    wells: int,
) -> None:
    holder_page.choose_plate(plate)

    view = holder_page.view()

    assert view.pose == MapPose(
        "estimated", holder_centre_um=HOLDER_CENTRE, a1_corner=STAGE_A1
    )
    assert view.nominal is True
    centres = _centres(view)
    assert len(centres) == wells
    assert centres["A1"] == pytest.approx(a1_um, abs=0.01)
    # The point the profile gives is the centre of the grid, whichever plate is on it.
    mean_x = sum(x for x, _ in centres.values()) / wells
    mean_y = sum(y for _, y in centres.values()) / wells
    assert (mean_x, mean_y) == pytest.approx(HOLDER_CENTRE, abs=0.01)


def test_the_estimate_of_demo_sample_is_its_own_plate(
    fake_core: FakeCore, run_doc: Path
) -> None:
    profile = Profile.load(PROFILES / "demo-sample.toml")
    assert profile.sample is not None
    for page, _ in _page(fake_core, run_doc, profile):
        page.choose_plate(profile.sample.plate)

        view = page.view()

        assert view.pose.kind == "estimated"
        centres = _centres(view)
        for ref in sample_refs(profile.sample):
            assert centres[ref.well] == pytest.approx((ref.x_um, ref.y_um), abs=0.01)
        # A1, A12 and H1: the corners the registration asks for.
        assert {ref.well for ref in sample_refs(profile.sample)} >= {"A1", "A12", "H1"}


def test_a_registered_plate_of_the_page_plate_wins_over_the_holder(
    holder_page: PlatePage, run_doc: Path
) -> None:
    _register(run_doc, _plate_result(a1=(1000.0, -500.0)))

    view = holder_page.view()

    assert view.pose == MapPose("registered", a1_corner=STAGE_A1)
    assert view.nominal is False
    assert _centres(view)["A1"] == pytest.approx((1000.0, -500.0), abs=0.01)


@pytest.mark.parametrize(
    ("refs", "well"),
    [
        ((WellRef(well="B3", x_um=10.0, y_um=20.0),), "B3"),
        ((), ""),
    ],
    ids=["one-reference", "no-reference"],
)
def test_a_seed_is_drawn_as_a_seed_and_names_its_well(
    holder_page: PlatePage,
    run_doc: Path,
    refs: tuple[WellRef, ...],
    well: str,
) -> None:
    _register(run_doc, _plate_result(method="seed", refs=refs))

    view = holder_page.view()

    assert view.pose == MapPose("seed", seed_well=well, a1_corner=STAGE_A1)
    assert view.nominal is False
    assert _centres(view)["A1"] == pytest.approx((1000.0, -500.0), abs=0.01)


@pytest.mark.parametrize("method", ["lattice-fit", "wall-images"])
def test_every_method_but_seed_is_a_registered_plate(
    holder_page: PlatePage, run_doc: Path, method: str
) -> None:
    _register(run_doc, _plate_result(method=method))

    assert holder_page.view().pose == MapPose("registered", a1_corner=STAGE_A1)


def test_a_plate_registered_for_another_plate_leaves_the_estimate(
    holder_page: PlatePage, run_doc: Path
) -> None:
    _register(run_doc, _plate_result(plate="96-well"))
    holder_page.choose_plate("greiner-24-well")

    view = holder_page.view()

    assert view.pose == MapPose(
        "estimated", holder_centre_um=HOLDER_CENTRE, a1_corner=STAGE_A1
    )
    assert view.nominal is True
    assert _centres(view)["A1"] == pytest.approx((-48_225.0, 28_850.0), abs=0.01)


def test_the_estimate_follows_the_plate_chosen_after_another_was_drawn(
    holder_page: PlatePage,
) -> None:
    # The grid's centre offset is kept per plate, and the first plate drawn must not give
    # its offset to the next: 875 µm and 2325 µm off for a 24-well after a 96-well, and
    # every well named by the wrong place. One page, three views, the plate chosen between.
    expected = {
        "96-well": (-49_100.0, 31_175.0),
        "greiner-24-well": (-48_225.0, 28_850.0),
    }
    for plate in ("96-well", "greiner-24-well", "96-well"):
        holder_page.choose_plate(plate)

        centres = _centres(holder_page.view())

        assert centres["A1"] == pytest.approx(expected[plate], abs=0.01)
        mean_x = sum(x for x, _ in centres.values()) / len(centres)
        mean_y = sum(y for _, y in centres.values()) / len(centres)
        assert (mean_x, mean_y) == pytest.approx(HOLDER_CENTRE, abs=0.01)


def test_a_click_reads_the_wells_the_view_draws_on_an_estimated_map(
    holder_page: PlatePage,
) -> None:
    # A select click on the estimate finds the well the map shows, in the estimate's
    # place (m5 §15 D6); a drive click goes there, below (m3 §5.5).
    centre = _centres(holder_page.view())["B3"]

    assert holder_page.click(*centre, drive=False) == "B3"
    # The stage origin is no longer where A1 is: it lies between wells on the estimate.
    assert holder_page.click(0.0, 0.0, drive=False) is None


# --- a drive on the estimate (m3 §5.5 D6, #462) --------------------------------------


def _where(microscope: Microscope) -> tuple[float, float]:
    xy = microscope.require(XYStage).position_um()
    return (xy.x_um, xy.y_um)


@pytest.mark.parametrize("orientation", ["not-turned", "turned-180"])
def test_the_estimated_map_and_the_drive_share_holder_estimate(
    holder_page: PlatePage, orientation: PlateOrientation
) -> None:
    holder_page.choose_plate("96-well")
    holder_page.choose_orientation(orientation)
    view = holder_page.view()
    assert view.pose.kind == "estimated"
    plan = holder_estimate("96-well", HOLDER_CENTRE, orientation)
    assert len(view.wells) == 96
    for well in view.wells:
        assert well.center_um == well_geometry(plan, well.well).center_um


def test_goto_on_an_estimated_map_passes_the_plate_and_the_orientation(
    holder_stand: tuple[PlatePage, Microscope], run_doc: Path
) -> None:
    page, microscope = holder_stand
    page.choose_plate("greiner-24-well")
    page.choose_orientation("turned-180")
    page.goto("B3")
    assert page.join(10.0)
    record = RunDocument.load(run_doc).stages["goto-well"]
    assert record.status is StageStatus.done
    assert record.params == {
        "well": "B3",
        "side": "centre",
        "plate": "greiner-24-well",
        "orientation": "turned-180",
    }
    # The Greiner 24-well's pitch is 19 450 µm, its grid 6 by 4: turned 180° about the
    # holder, B3 lies where C4 does on the plate not turned, written here by hand.
    assert _where(microscope) == pytest.approx((10_125.0, -10_050.0), abs=0.01)
    assert page.status().goto.startswith("goto-well done in ")


def test_goto_on_an_unplaced_map_is_refused_before_anything_starts(
    bare_stand: tuple[PlatePage, Microscope], run_doc: Path
) -> None:
    page, microscope = bare_stand
    page.choose_plate("96-well")
    before = _where(microscope)
    with pytest.raises(ValueError, match=f"^{re.escape(GOTO_UNPLACED)}$"):
        page.goto("A1")
    assert page.status().goto == GOTO_UNPLACED
    assert page.status().running is None
    assert page.current is None
    assert "goto-well" not in RunDocument.load_or_new(run_doc).stages
    assert _where(microscope) == before


def test_goto_on_the_placeholder_plate_is_refused_with_plate_not_chosen(
    holder_stand: tuple[PlatePage, Microscope], run_doc: Path
) -> None:
    page, microscope = holder_stand
    assert not page.plate_chosen  # the demo profile names no plate: the placeholder
    before = _where(microscope)
    with pytest.raises(ValueError, match=f"^{re.escape(PLATE_NOT_CHOSEN)}$"):
        page.goto("A1")
    assert page.status().goto == PLATE_NOT_CHOSEN
    assert "goto-well" not in RunDocument.load_or_new(run_doc).stages
    assert _where(microscope) == before


def test_goto_to_a_wall_on_an_estimated_map_is_refused_before_anything_starts(
    holder_stand: tuple[PlatePage, Microscope], run_doc: Path
) -> None:
    page, microscope = holder_stand
    page.choose_plate("96-well")
    before = _where(microscope)
    with pytest.raises(
        ValueError, match=f"^{re.escape(GOTO_WALL_NEEDS_REGISTRATION)}$"
    ):
        page.goto("B2", "left")
    assert page.status().goto == GOTO_WALL_NEEDS_REGISTRATION
    assert page.current is None
    assert "goto-well" not in RunDocument.load_or_new(run_doc).stages
    assert _where(microscope) == before


def test_goto_on_a_registered_plate_passes_todays_parameters(
    holder_page: PlatePage, run_doc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register(run_doc, _plate_result(plate="96-well"))
    holder_page.choose_plate("96-well")
    runs: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        holder_page, "run", lambda plugin, params: runs.append((plugin, list(params)))
    )
    holder_page.goto("B2")
    assert runs == [("goto-well", ["well=B2", "side=centre"])]
    assert holder_page.status().goto == "going to B2"
