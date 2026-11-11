"""Where the map draws the plate before and after a registration: its four poses (m5 §15 D3, #410).

``PlateView.pose`` says which of *registered*, *seed*, *estimated* and *unplaced* the map is, and
the wells are drawn where that pose puts them. These tests drive the controller alone (no Panel,
no browser), on the facade over ``FakeCore`` with a profile that does or does not carry a
``[holder]``. The 96-well plate's grid is not square (12 columns by 8 rows), so swapped axes or a
flipped sign show (FM-73); the Greiner 24-well is the lab's own plate (#400).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.profile import HolderSection, Profile
from smc.plugins.slots import PlateResult, WellRef
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.testing.sample_stand import sample_refs
from smc.ui.controller import MapPose, PlatePage, PlateView

PROFILES = Path(__file__).resolve().parents[2] / "profiles"
HOLDER_CENTRE = (400.0, -325.0)
MEASURED = "2026-11-10, sessions 2 and 3 (#334, #405): the mean of two centres"
THEN = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
#: A1's corner on a stand that does not say its sides (m5 §16 D3): the demo's.
STAGE_A1 = "stage -X +Y"


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
def holder_page(fake_core: FakeCore, run_doc: Path) -> Iterator[PlatePage]:
    for page, _ in _page(fake_core, run_doc, _with_holder()):
        yield page


@pytest.fixture
def bare_page(fake_core: FakeCore, run_doc: Path) -> Iterator[PlatePage]:
    for page, _ in _page(fake_core, run_doc, _with_holder(None)):
        yield page


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
    # The estimate is drawn and never acted on (m5 §15 D6): a select click on it finds the
    # well the map shows, in the estimate's place, and a drive click moves nothing.
    centre = _centres(holder_page.view())["B3"]

    assert holder_page.click(*centre, drive=False) == "B3"
    # The stage origin is no longer where A1 is: it lies between wells on the estimate.
    assert holder_page.click(0.0, 0.0, drive=False) is None
