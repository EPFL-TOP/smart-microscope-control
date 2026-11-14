"""The plate page's offsets block, in the controller: the card, the offer, the refusals, the step, the confirmed run, the plan and the write (m5 §18, #382).

No Panel here. The page is the real controller over ``fake_microscope`` with the real
``objective-offsets`` loaded from its own entry-point value, never from the installed
group (FM-112); the run document is built by ``write_stage`` from records written out
by hand, and every expected text is the one of the design, asserted whole (FM-45).
Nothing here turns a real turret: the step and the write are read and file work, and
the confirmed run's comparison is tested by calling the controller with what it would
refuse (FM-43, FM-169).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins.errors import PluginRunError
from smc.plugins.objective_offsets.plugin import _refuse_wells_that_disagree
from smc.plugins.registry import GROUP
from smc.plugins.slots import (
    MeasuredOffset,
    ObjectiveOffsetsResult,
    OffsetWell,
    PlateResult,
    TargetMeasure,
    WallImages,
    WellFit,
)
from smc.run import ReadRecord, RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import NOT_CONNECTED, PlatePage
from smc.ui.guide import OFFSETS_DRY_RUN, offsets_plate_refusal

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
OFFSETS = "smc.plugins.objective_offsets.plugin:ObjectiveOffsetsPlugin"
WALLS = "smc.plugins.plate_walls.plugin:PlateWallsPlugin"
REGISTRATION = "smc.plugins.plate_registration.plugin:PlateRegistrationPlugin"


def _entry_points(*names: str) -> list[EntryPoint]:
    values = {
        "objective-offsets": OFFSETS,
        "plate-walls": WALLS,
        "plate-registration": REGISTRATION,
    }
    return [EntryPoint(name=n, value=values[n], group=GROUP) for n in names]


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


# --- records written out by hand ----------------------------------------------------


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


def _walls_plate(
    *,
    objective: str | None = TEN,
    walls_objective: str | None = TEN,
    method: str = "wall-images",
    with_walls: bool = True,
    plate: str = "96-well",
) -> PlateResult:
    """A plate as ``plate-walls`` leaves it (``method='wall-images'``) unless told otherwise."""
    walls = WallImages(
        objective=walls_objective,
        pixel_size_um=1.1,
        image_shape_px=(100, 100),
        search_um=1000.0,
        wells=(_fit("A1", (0.0, 0.0)), _fit("H12", (99000.0, -63000.0))),
        frames=8,
        folder=None,
        elapsed_s=10.0,
    )
    return PlateResult(
        plate=plate,
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method=method,  # type: ignore[arg-type]
        residual_um=1.0,
        scale_error=None,
        n_refs=2,
        walls=walls if with_walls else None,
        objective=objective,
    )


def _measure(
    objective: str,
    shift: tuple[float, float],
    *,
    wells: tuple[str, ...] = ("A1", "H12", "A12"),
    spread: float = 0.03,
    bound: float = 2.0,
    folder: str | None = None,
) -> TargetMeasure:
    return TargetMeasure(
        objective=objective,
        calibration="cal.json",
        calibration_measured="2026-11-10T10:00:00+00:00",
        pixel_size_um=2.5,
        z_after_turn_um=None,
        wells=tuple(
            OffsetWell(
                well=well,
                origin_xy_um=(0.0, 0.0),
                delta_xy_um=shift,
                fit=_fit(well, shift),
            )
            for well in wells
        ),
        shift_xy_um=shift,
        spread_um=spread,
        max_spread_um=bound,
        frames=24,
        folder=folder,
    )


def _result(
    *targets: tuple[str, tuple[float, float]], plate_objective: str = TEN
) -> ObjectiveOffsetsResult:
    """A result of the plate registered under ``plate_objective``, each target's offset its own shift."""
    return ObjectiveOffsetsResult(
        plate_objective=plate_objective,
        reference=plate_objective,
        reference_in_profile=False,
        measures=tuple(_measure(label, shift) for label, shift in targets),
        offsets=tuple(
            MeasuredOffset(objective=label, xy_offset_um=shift)
            for label, shift in targets
        ),
        objective_in_place=targets[-1][0],
        elapsed_s=62.8,
    )


def _seed(
    run_doc: Path,
    plugin: str,
    provides: str,
    result: dict[str, Any] | None = None,
    *,
    status: StageStatus = StageStatus.done,
    minute: int = 0,
    profile: str = "demo",
    dry_run: bool = False,
    error: str | None = None,
    params: dict[str, Any] | None = None,
    reads: dict[str, ReadRecord | None] | None = None,
) -> None:
    """A record an earlier run of ``plugin`` left; ``minute`` orders the starts and names the attempt."""
    then = T0 + timedelta(minutes=minute)
    write_stage(
        run_doc,
        StageRecord(
            plugin=plugin,
            provides=provides,
            attempt=f"a{minute}",
            reads=reads or {},
            status=status,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=dry_run,
            smc_version="0",
            params=params if params is not None else {},
            result=result,
            error=error,
        ),
    )


def _seed_plate(
    run_doc: Path,
    plate: PlateResult | None = None,
    *,
    plugin: str = "plate-walls",
    minute: int = 0,
) -> None:
    found = plate or _walls_plate()
    _seed(
        run_doc,
        plugin,
        "plate",
        found.model_dump(mode="json"),
        minute=minute,
        params={"plate": found.plate},
    )


def _seed_offsets(
    run_doc: Path,
    result: ObjectiveOffsetsResult | None = None,
    *,
    plate_plugin: str = "plate-walls",
    minute: int = 1,
    status: StageStatus = StageStatus.done,
    error: str | None = None,
) -> None:
    """The record an earlier run of ``objective-offsets`` left, reading the plate seeded at minute 0."""
    _seed(
        run_doc,
        "objective-offsets",
        "objective-offsets",
        None if result is None else result.model_dump(mode="json"),
        status=status,
        minute=minute,
        error=error,
        reads={"plate": ReadRecord(plugin=plate_plugin, attempt="a0")},
    )


def _page(microscope: Microscope, run_doc: Path, *names: str) -> PlatePage:
    return PlatePage(
        microscope,
        run_doc,
        entry_points=_entry_points("objective-offsets", *names),
    )


def _dry_microscope(fake_core: FakeCore) -> Microscope:
    return Microscope.from_core(
        cast("CMMCorePlus", fake_core), Profile.demo(), dry_run=True
    )


# --- the card (D6) ----------------------------------------------------------------


def test_the_card_of_an_offsets_result_names_each_target_and_not_item_counts(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    # Asymmetric shifts and a -0.0, written out from the numbers (FM-203).
    _seed_offsets(run_doc, _result((TWENTY, (-5.02, 2.49)), (FORTY, (-0.0, 12.25))))
    page = _page(fake_microscope, run_doc)
    try:
        card = page.view().cards["objective-offsets"]
    finally:
        page.close()

    assert card.lines == (
        f"{TWENTY}: shift from {TEN} (-5.02, +2.49) µm, spread 0.03 µm over A1, H12, "
        "A12 (bound 2.00 µm)",
        f"{FORTY}: shift from {TEN} (+0.00, +12.25) µm, spread 0.03 µm over A1, H12, "
        "A12 (bound 2.00 µm)",
        f"offsets to write: {TWENTY} (-5.02, +2.49) µm, {FORTY} (+0.00, +12.25) µm",
        f"the turret is left at {FORTY}",
    )
    assert not any("items" in line for line in card.lines)


def test_the_cards_lines_are_one_line_each_and_bounded_for_a_label_of_700_characters(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # FM-205: the longest form of a line that carries a label, through the card builder.
    label = "40X\nlens " + "x" * 700
    _seed_plate(run_doc)
    _seed_offsets(run_doc, _result((label, (1.0, 2.0))))
    page = _page(fake_microscope, run_doc)
    try:
        card = page.view().cards["objective-offsets"]
    finally:
        page.close()

    assert len(card.lines) == 3
    assert all("\n" not in line for line in card.lines)
    assert all(len(line) == 601 and line.endswith("…") for line in card.lines)


def _five_targets_and_six_wells() -> str:
    """The tool's own refusal of wells that disagree, for the longest run it takes: five targets, six wells each."""
    wells = ("A1", "H12", "A12", "H1", "D6", "B2")
    labels = (
        "1-Plan Fluor 4x",
        "2-Plan Fluor 10x",
        "3-Plan Apo LmbdD0.25 40x",
        "Nikon 20X Plan Fluor ELWD",
        "Nikon 60X Plan Apo",
    )
    measures = [
        _measure(
            label,
            (-1234.5 - n, 987.25 + n),
            wells=wells,
            spread=9.87,
            bound=5.5,
            folder=f"objective-offsets/0123456789abcdef/{n}",
        )
        for n, label in enumerate(labels, start=1)
    ]
    with pytest.raises(PluginRunError) as raised:
        _refuse_wells_that_disagree(measures)
    return str(raised.value)


def test_the_card_keeps_a_refusal_of_five_targets_and_six_wells_whole(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    refusal = _five_targets_and_six_wells()
    # The case that bound was chosen for: longer than a diagnosis's 600, under 2000.
    assert 600 < len(refusal) < 2000
    _seed_plate(run_doc)
    _seed_offsets(
        run_doc,
        status=StageStatus.failed,
        error=f"PluginRunError: {refusal}",
    )
    page = _page(fake_microscope, run_doc)
    try:
        card = page.view().cards["objective-offsets"]
    finally:
        page.close()

    assert card.error == f"PluginRunError: {refusal}"


def test_an_error_of_another_tool_is_still_cut_at_600_and_this_ones_at_2000(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    long = "x" * 2500
    _seed_plate(run_doc)
    _seed_offsets(
        run_doc,
        status=StageStatus.failed,
        error=f"PluginRunError: objective-offsets: {long}",
    )
    # A tool the page also loads, whose record is `failed` with the same long text.
    _seed(
        run_doc,
        "plate-walls",
        "plate",
        None,
        status=StageStatus.failed,
        minute=2,
        error=f"PluginRunError: plate-walls: {long}",
    )
    page = _page(fake_microscope, run_doc, "plate-walls")
    try:
        cards = page.view().cards
    finally:
        page.close()

    offsets_error = cards["objective-offsets"].error
    walls_error = cards["plate-walls"].error
    assert offsets_error is not None
    assert walls_error is not None
    assert len(offsets_error) == 2001
    assert offsets_error.endswith("…")
    assert len(walls_error) == 601
    assert walls_error.endswith("…")


# --- the offer (D2) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "offered"),
    [
        ("walls", True),
        ("walls-no-objective", False),
        ("registration-by-eye", False),
        ("plate-finding", False),
        ("no-plate", False),
        ("another-page-plate", False),
        ("mixed-objectives", True),
    ],
)
def test_the_offer_is_made_for_a_plate_registered_from_its_walls_with_its_objective_and_for_none_other(
    fake_microscope: Microscope, run_doc: Path, case: str, offered: bool
) -> None:
    plates = {
        "walls": _walls_plate(),
        "walls-no-objective": _walls_plate(objective=None, walls_objective=None),
        "registration-by-eye": _walls_plate(
            method="references", with_walls=False, walls_objective=None
        ),
        "plate-finding": _walls_plate(
            method="lattice-fit", with_walls=False, walls_objective=None
        ),
        "another-page-plate": _walls_plate(plate="24-well"),
        # A hand-mixed record: offered, and the block's wait says what is wrong with it (D8).
        "mixed-objectives": _walls_plate(walls_objective=FORTY),
    }
    if case != "no-plate":
        _seed_plate(run_doc, plates[case])
    page = _page(fake_microscope, run_doc)
    try:
        if case == "another-page-plate":
            # The page's plate is another one than the registered plate's.
            page.choose_plate("6-well")
        offer = page.view().offsets
    finally:
        page.close()

    if offered:
        assert offer is not None
        assert offer.plate_objective == TEN
    else:
        assert offer is None


# --- the refusals the page foresees (D8, FM-206) ----------------------------------


def _bytes_and_provider(run_doc: Path) -> tuple[bytes, StageRecord | None]:
    return run_doc.read_bytes(), RunDocument.load_or_new(run_doc).provider(
        "objective-offsets"
    )


def test_a_dry_run_page_refuses_objective_offsets_in_the_controller_and_writes_nothing(
    fake_core: FakeCore, run_doc: Path
) -> None:
    # A usable result of the tool, as a real run left it: the tool's own refusal of a
    # dry run would be a `failed` record that replaces it (§11.4, FM-206).
    _seed_plate(run_doc)
    _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25))))
    dry = _dry_microscope(fake_core)
    page = _page(dry, run_doc)
    try:
        before, provider = _bytes_and_provider(run_doc)
        assert provider is not None
        assert provider.status is StageStatus.done
        # Pairs that the parameters would refuse: the dry run's refusal comes first.
        page.run("objective-offsets", ["bogus=1"])
        outcome = page.status().outcome
        running = page.status().running
        after, again = _bytes_and_provider(run_doc)
    finally:
        page.close()
        dry.close()

    assert running is None
    assert outcome == f"Objective offsets: {OFFSETS_DRY_RUN}"
    assert outcome == (
        "Objective offsets: a dry run cannot turn the turret or image the walls; "
        "run it without --dry-run"
    )
    assert after == before
    assert again is not None
    assert (again.attempt, again.status) == (provider.attempt, StageStatus.done)


@pytest.mark.parametrize("case", ["by-eye", "no-objective", "mixed-objectives"])
def test_a_plate_that_is_not_from_the_walls_refuses_objective_offsets_in_the_controller_and_writes_nothing(
    fake_microscope: Microscope, run_doc: Path, case: str
) -> None:
    plates = {
        "by-eye": _walls_plate(
            method="references", with_walls=False, walls_objective=None
        ),
        "no-objective": _walls_plate(objective=None),
        "mixed-objectives": _walls_plate(walls_objective=FORTY),
    }
    plate = plates[case]
    expected = offsets_plate_refusal(plate)
    assert expected is not None
    _seed_plate(run_doc, plate, plugin="plate-registration")
    _seed_offsets(
        run_doc, _result((FORTY, (-37.5, 12.25))), plate_plugin="plate-registration"
    )
    page = _page(fake_microscope, run_doc)
    try:
        before, provider = _bytes_and_provider(run_doc)
        assert provider is not None
        assert provider.status is StageStatus.done
        page.run("objective-offsets", ["bogus=1"])
        outcome = page.status().outcome
        running = page.status().running
        after, again = _bytes_and_provider(run_doc)
    finally:
        page.close()

    assert running is None
    assert outcome == f"Objective offsets: {expected}"
    assert after == before
    assert again is not None
    assert (again.attempt, again.status) == (provider.attempt, StageStatus.done)


def test_a_plate_the_tool_can_measure_from_is_not_refused_by_the_page(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    # The control of the tests above: the page's refusal is the plate's, not the tool's own
    # parameters (`bogus` is the parameters' refusal, which comes after the page's).
    _seed_plate(run_doc)
    page = _page(fake_microscope, run_doc)
    try:
        page.run("objective-offsets", ["bogus=1"])
        outcome = page.status().outcome or ""
    finally:
        page.close()

    assert outcome.startswith("objective-offsets: ParamsError")
    assert "bogus" in outcome


def test_objective_offsets_is_blocked_by_the_dry_run_before_what_it_reads_and_by_the_plate_after_it(
    fake_microscope: Microscope, fake_core: FakeCore, run_doc: Path
) -> None:
    def blocked(microscope: Microscope) -> str | None:
        page = _page(microscope, run_doc)
        try:
            return page.view().blocked.get("objective-offsets")
        finally:
            page.close()

    dry = _dry_microscope(fake_core)
    try:
        # A dry run, and no plate at all: what the tool reads is missing, and the dry run is
        # the reason that holds longer, so it comes first.
        assert blocked(dry) == OFFSETS_DRY_RUN
        no_plate = blocked(fake_microscope)
        assert no_plate is not None
        assert no_plate.startswith("the run document has no plate")
        # A plate registered by eye: a real run is refused for it, a dry run for the dry run.
        eye = _walls_plate(method="references", with_walls=False, walls_objective=None)
        _seed_plate(run_doc, eye, plugin="plate-registration")
        assert blocked(dry) == OFFSETS_DRY_RUN
        assert blocked(fake_microscope) == offsets_plate_refusal(eye)
        # A plate from the walls: nothing the page foresees.
        _seed_plate(run_doc, _walls_plate(), minute=1)
        assert blocked(fake_microscope) is None
    finally:
        dry.close()

    closed = PlatePage.closed(
        fake_microscope.profile,
        run_doc,
        entry_points=_entry_points("objective-offsets"),
    )
    try:
        assert closed.view().blocked["objective-offsets"] == NOT_CONNECTED
    finally:
        closed.close()
