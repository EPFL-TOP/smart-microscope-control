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

import json
import os
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.ui.controller as controller_module
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.profile import Profile
from smc.hardware.profile_write import BLOCK_BEGIN, BLOCK_END
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.objective_offsets.plugin import _refuse_wells_that_disagree
from smc.plugins.params import parse_params
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
from smc.ui.controller import (
    NOT_CONNECTED,
    CalibrationRead,
    OffsetsPlanView,
    PlatePage,
)
from smc.ui.guide import (
    OFFSETS_DRY_RUN,
    OFFSETS_NO_DOCUMENT,
    OFFSETS_RESTART_FIRST,
    OFFSETS_TOO_MANY,
    OFFSETS_TURRET_UNREAD,
    CalibrationNumbers,
    busy_note,
    offsets_no_objective,
    offsets_plate_refusal,
)

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
    *targets: tuple[str, tuple[float, float]],
    plate_objective: str = TEN,
    reference_in_profile: bool = False,
) -> ObjectiveOffsetsResult:
    """A result of the plate registered under ``plate_objective``, each target's offset its own shift."""
    return ObjectiveOffsetsResult(
        plate_objective=plate_objective,
        reference=plate_objective,
        reference_in_profile=reference_in_profile,
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


# --- the step (D3, D8) and the confirmed run (D4) -----------------------------------


class StubOffsetsParams(PluginParams):
    """The tool's three parameters the block fills, as the real ones are."""

    targets: tuple[str, ...]
    calibrations: tuple[Path, ...]
    confirm_turns: bool = False


class StubOffsets:
    """``objective-offsets`` without a stage: it keeps what it was given and returns a result of the targets."""

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
        return _result(*((label, (1.0, 2.0)) for label in params.targets))


class StubSlow(PluginParams):
    pass


class SlowResult(PluginResult):
    pass


class Slow:
    """A tool that waits until the test lets it go: an activity that is going."""

    name = provides = "slow"
    summary = "A test tool that waits."
    Params = StubSlow
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
    Slow.started = threading.Event()
    Slow.release = threading.Event()
    try:
        yield
    finally:
        Slow.release.set()  # a worker a failed test left behind ends


def _stub_page(microscope: Microscope, run_doc: Path) -> PlatePage:
    return PlatePage(
        microscope,
        run_doc,
        entry_points=[
            EntryPoint(name=c.name, value=f"{__name__}:{c.__name__}", group=GROUP)
            for c in (StubOffsets, Slow)
        ],
    )


def _document(
    folder: Path,
    objective: str,
    seed: Callable[..., Any],
    *,
    name: str | None = None,
    profile: str = "demo",
) -> Path:
    """A run document holding the camera calibration of ``objective``, in a file of its own."""
    path = folder / (name or f"cal-{objective.split()[1].lower()}.json")
    seed(path, objective=objective, profile=profile)
    return path


def _numbers_written_in(path: Path) -> CalibrationNumbers:
    """The numbers of the calibration a seeded document holds, read back from its own JSON."""
    record = RunDocument.load(path).stages["camera-calibration"]
    assert record.result is not None
    result = record.result
    return CalibrationNumbers(
        pixel_size_um=result["pixel_size_um"],
        rotation_deg=result["rotation_deg"],
        mirrored=result["mirrored"],
        residual_um=result["residual_um"],
        step_um=result["step_um"],
    )


def _typed(*paths: Path | str) -> str:
    """What an operator typed: one path per line, as they are."""
    return "\n".join(str(path) for path in paths)


def _turret_labels(microscope: Microscope) -> tuple[str, ...]:
    return tuple(
        label for label in microscope.require(ObjectiveTurret).labels() if label
    )


def test_offsets_step_reads_each_document_and_derives_the_objectives_in_line_order(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    forty = _document(tmp_path, FORTY, seed_camera_calibration)
    twenty = _document(tmp_path, TWENTY, seed_camera_calibration)
    page = _stub_page(fake_microscope, run_doc)
    try:
        # 40X first, then 20X: the order the turret turns in, the documents' and not the
        # turret's (the 20X is the turret's fourth position, the 40X its first).
        step = page.offsets_step(_typed(twenty, forty))
        backwards = page.offsets_step(_typed(forty, twenty))
    finally:
        page.close()

    assert step.wait is None
    assert step.targets == (TWENTY, FORTY)
    assert backwards.targets == (FORTY, TWENTY)
    when = "2026-11-09T09:30:00+00:00"
    assert step.reads == (
        # The frame the seeded calibration was measured in is read too (#617: the scan plan's
        # click compares it with the camera's), and its numbers (#627, m5 §23.2 D4: the Stand
        # screen's pane), so the read is not equal to one without them. The numbers are the
        # ones the document holds, read back from the file the test wrote (FM-203).
        CalibrationRead(
            str(twenty),
            TWENTY,
            when,
            None,
            image_shape_px=(384, 512),
            numbers=_numbers_written_in(twenty),
        ),
        CalibrationRead(
            str(forty),
            FORTY,
            when,
            None,
            image_shape_px=(384, 512),
            numbers=_numbers_written_in(forty),
        ),
    )
    assert _numbers_written_in(forty) == CalibrationNumbers(
        pixel_size_um=1.1,
        rotation_deg=180.0,
        mirrored=False,
        residual_um=0.01,
        step_um=48.0,
    )
    # The pairs are what `smc run objective-offsets` is given, read back by the parameters'
    # own parser, never compared as strings (FM-45).
    parsed = parse_params(StubOffsetsParams, [*step.pairs, "confirm_turns=true"])
    assert parsed.targets == (TWENTY, FORTY)  # type: ignore[attr-defined]
    assert parsed.calibrations == (twenty, forty)  # type: ignore[attr-defined]
    assert parsed.confirm_turns is True  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "case", ["missing", "not-json", "no-calibration", "other-profile", "folder"]
)
def test_offsets_step_waits_on_the_first_document_the_tool_refuses_and_says_its_words(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    case: str,
) -> None:
    _seed_plate(run_doc)
    good = _document(tmp_path, FORTY, seed_camera_calibration)
    bad = tmp_path / "bad.json"
    where = bad.as_posix()
    expected: dict[str, str] = {
        "missing": f"{where}: does not exist",
        "no-calibration": (
            f"in {where}: the run document has no camera-calibration; smc run --list "
            "shows which plugin provides it"
        ),
        "other-profile": (
            f"in {where}: camera-calibration was run on lightsheet, this is demo; run "
            "camera-calibration again on this profile"
        ),
        "folder": f"{where}: is a folder, not a run document; name a file",
    }
    if case == "not-json":
        bad.write_text("{not json", encoding="utf-8")
    elif case == "no-calibration":
        _seed(bad, "plate-walls", "plate", _walls_plate().model_dump(mode="json"))
    elif case == "other-profile":
        seed_camera_calibration(bad, objective=TWENTY, profile="lightsheet")
    elif case == "folder":
        bad.mkdir()
    page = _stub_page(fake_microscope, run_doc)
    try:
        # The good document first and another one after: the first with a problem is the wait.
        step = page.offsets_step(_typed(good, bad, tmp_path / "also-missing.json"))
    finally:
        page.close()

    assert step.pairs == ()
    assert step.wait is not None
    if case == "not-json":
        assert step.wait.startswith(f"{where}: is not valid JSON: ")
    else:
        assert step.wait == expected[case]
    assert step.reads[0].problem is None
    assert step.reads[1].problem == step.wait
    assert step.targets == (FORTY,)


def test_offsets_step_waits_for_what_the_page_holds_in_its_order(
    fake_microscope: Microscope,
    fake_core: FakeCore,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    forty = _document(tmp_path, FORTY, seed_camera_calibration)
    text = _typed(forty)
    # No plate at all: what the tool reads is missing (the reads' own reason).
    page = _stub_page(fake_microscope, run_doc)
    try:
        no_plate = page.offsets_step(text).wait
        _seed_plate(run_doc)
        assert page.offsets_step("").wait == OFFSETS_NO_DOCUMENT
        assert page.offsets_step(" \n\n").wait == OFFSETS_NO_DOCUMENT
        # A run going: the activity's name, said as the tools' own waits say it.
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        going = page.offsets_step(text).wait
        Slow.release.set()
        assert page.join(10.0)
        # The run document cannot be read: not an empty one (FM-229).
        run_doc.write_text("{not json", encoding="utf-8")
        unreadable = page.offsets_step(text).wait
    finally:
        page.close()
    dry = _dry_microscope(fake_core)
    dry_page = _stub_page(dry, run_doc)
    try:
        run_doc.unlink()
        # A dry run: its words come first, before the plate the page cannot read.
        dry_wait = dry_page.offsets_step(text).wait
    finally:
        dry_page.close()
        dry.close()
    closed = PlatePage.closed(
        fake_microscope.profile,
        run_doc,
        entry_points=[EntryPoint(name="objective-offsets", value=OFFSETS, group=GROUP)],
    )
    try:
        closed_wait = closed.offsets_step(text).wait
    finally:
        closed.close()

    assert no_plate is not None
    assert no_plate.startswith("the run document has no plate")
    assert going == busy_note("slow", live=False)
    assert unreadable is not None
    assert unreadable.startswith("run document: ")
    assert dry_wait == OFFSETS_DRY_RUN
    assert closed_wait == NOT_CONNECTED


def test_offsets_step_refuses_the_plates_own_objective_one_not_on_the_turret_and_one_given_twice(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    labels = _turret_labels(fake_microscope)
    ten = _document(tmp_path, TEN, seed_camera_calibration)
    sixty = _document(tmp_path, "Nikon 60X Plan Apo", seed_camera_calibration)
    forty = _document(tmp_path, FORTY, seed_camera_calibration)
    forty_again = _document(
        tmp_path, FORTY, seed_camera_calibration, name="cal-40x-again.json"
    )
    twenty = _document(tmp_path, TWENTY, seed_camera_calibration)
    page = _stub_page(fake_microscope, run_doc)
    try:
        own = page.offsets_step(_typed(forty, ten)).wait
        off_turret = page.offsets_step(_typed(forty, sixty)).wait
        twice = page.offsets_step(_typed(forty, twenty, forty_again)).wait
        # The first that holds in line order: the 60X comes before the plate's own 10X.
        first = page.offsets_step(_typed(sixty, ten)).wait
        fine = page.offsets_step(_typed(forty, twenty)).wait
    finally:
        page.close()

    assert own == (
        f"{TEN} is the objective the plate was registered with: its offset from "
        "itself is zero; leave its document out"
    )
    assert off_turret == f"Nikon 60X Plan Apo is not on the turret: {', '.join(labels)}"
    assert twice == f"{FORTY} is in two documents: {forty} and {forty_again}"
    assert first == f"Nikon 60X Plan Apo is not on the turret: {', '.join(labels)}"
    assert fine is None


def test_offsets_step_reads_no_sixth_line(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    # The demo turret has exactly five objectives besides the plate's 10X.
    others = [label for label in _turret_labels(fake_microscope) if label != TEN]
    assert len(others) == 5
    folder = tmp_path / "docs"
    folder.mkdir()
    documents = [
        _document(folder, label, seed_camera_calibration, name=f"cal-{n}.json")
        for n, label in enumerate(others)
    ]
    sixth = tmp_path / "sixth.json"  # not there: read, it would say so
    page = _stub_page(fake_microscope, run_doc)
    try:
        step = page.offsets_step(_typed(*documents, sixth))
        five = page.offsets_step(_typed(*documents))
    finally:
        page.close()

    assert len(step.reads) == 5
    assert len(step.targets) == 5
    assert step.pairs == ()
    assert step.wait == OFFSETS_TOO_MANY
    assert five.wait is None
    assert len(five.pairs) == 2


def test_offsets_step_encodes_the_paths_as_json_so_a_windows_path_survives(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    # Backslashes in the typed text: a Windows path as Copy as path gives it (in quotes),
    # or, on another OS, a file name that holds backslashes. Either way the file is there.
    typed = f"{tmp_path}{os.sep}docs\\cal-40x.json"
    Path(typed).parent.mkdir(parents=True, exist_ok=True)
    seed_camera_calibration(Path(typed), objective=FORTY)
    page = _stub_page(fake_microscope, run_doc)
    try:
        step = page.offsets_step(f'"{typed}"')
    finally:
        page.close()

    assert step.wait is None
    assert step.reads[0].path == typed
    # The quotes were dropped and the path is as typed, doubled by the encoder only.
    calibrations = step.pairs[1]
    assert calibrations.startswith("calibrations=[")
    assert json.loads(calibrations.partition("=")[2]) == [typed]
    assert "docs\\\\cal-40x.json" in calibrations
    assert parse_params(StubOffsetsParams, list(step.pairs)).calibrations == (  # type: ignore[attr-defined]
        Path(typed),
    )


def _documents_for(tmp_path: Path, seed: Callable[..., Any]) -> tuple[Path, Path]:
    return (
        _document(tmp_path, FORTY, seed),
        _document(tmp_path, TWENTY, seed),
    )


def test_measure_offsets_sends_confirm_turns_only_when_the_armed_objectives_are_still_the_documents(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    forty, twenty = _documents_for(tmp_path, seed_camera_calibration)
    text = _typed(forty, twenty)
    page = _stub_page(fake_microscope, run_doc)
    try:
        armed = page.offsets_step(text).targets
        assert armed == (FORTY, TWENTY)
        page.measure_offsets(armed, text)
        assert page.join(10.0)
        outcome = page.status().outcome
        record = RunDocument.load_or_new(run_doc).provider("objective-offsets")
    finally:
        page.close()

    assert [(p.targets, p.calibrations, p.confirm_turns) for p in StubOffsets.seen] == [
        ((FORTY, TWENTY), (forty, twenty), True)
    ]
    assert record is not None
    assert record.status is StageStatus.done
    assert record.params["confirm_turns"] is True
    assert outcome is not None
    assert outcome.startswith("Objective offsets done in ")


@pytest.mark.parametrize(
    "armed",
    [(TWENTY, FORTY), (FORTY,), (FORTY, TWENTY, "Objective-2"), (TEN,)],
    ids=["reordered", "one-short", "one-more", "another"],
)
def test_measure_offsets_with_other_armed_objectives_turns_nothing_and_says_what_changed(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    armed: tuple[str, ...],
) -> None:
    _seed_plate(run_doc)
    forty, twenty = _documents_for(tmp_path, seed_camera_calibration)
    text = _typed(forty, twenty)
    page = _stub_page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        page.measure_offsets(armed, text)
        running = page.status().running
        outcome = page.status().outcome
        after = run_doc.read_bytes()
    finally:
        page.close()

    assert StubOffsets.seen == []
    assert running is None
    assert after == before
    assert outcome == (
        "Objective offsets: the objectives changed since you confirmed them "
        f"({', '.join(armed)} → {FORTY}, {TWENTY}): measure again"
    )


def test_measure_offsets_with_changed_documents_turns_nothing_and_says_what_changed(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    forty, twenty = _documents_for(tmp_path, seed_camera_calibration)
    text = _typed(forty, twenty)
    page = _stub_page(fake_microscope, run_doc)
    try:
        armed = page.offsets_step(text).targets
        # Between the question and the click the second file is replaced by the calibration of
        # another objective: the turret would go to an objective the operator did not confirm.
        seed_camera_calibration(twenty, objective="Objective-2")
        before = run_doc.read_bytes()
        page.measure_offsets(armed, text)
        running = page.status().running
        outcome = page.status().outcome
        after = run_doc.read_bytes()
        # Or the file is taken away: the document's own words, nothing turns.
        twenty.unlink()
        page.measure_offsets(armed, text)
        gone = page.status().outcome
    finally:
        page.close()

    assert StubOffsets.seen == []
    assert running is None
    assert after == before
    assert outcome == (
        "Objective offsets: the objectives changed since you confirmed them "
        f"({FORTY}, {TWENTY} → {FORTY}, Objective-2): measure again"
    )
    assert gone == f"Objective offsets: {twenty.as_posix()}: does not exist"


def test_a_click_on_turn_and_measure_with_nothing_armed_turns_nothing_in_the_controller(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    forty, _ = _documents_for(tmp_path, seed_camera_calibration)
    page = _stub_page(fake_microscope, run_doc)
    try:
        before = run_doc.read_bytes()
        page.measure_offsets((), _typed(forty))
        outcome = page.status().outcome
        running = page.status().running
        after = run_doc.read_bytes()
    finally:
        page.close()

    assert StubOffsets.seen == []
    assert running is None
    assert after == before
    assert outcome == (
        "Objective offsets: choose the documents and press Measure the offsets again"
    )


def test_measure_offsets_is_refused_while_an_activity_goes(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
) -> None:
    _seed_plate(run_doc)
    forty, _ = _documents_for(tmp_path, seed_camera_calibration)
    text = _typed(forty)
    page = _stub_page(fake_microscope, run_doc)
    try:
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        with pytest.raises(RuntimeError, match=r"^slow is running$"):
            page.measure_offsets((FORTY,), text)
    finally:
        Slow.release.set()
        assert page.join(10.0)
        page.close()

    assert StubOffsets.seen == []


def test_nothing_else_in_the_page_writes_confirm_turns() -> None:
    # The turn is confirmed by one click on a question that names what turns (FM-14): the
    # block's second click is the only code that sends the pair. The accordion's own form has
    # its checkbox, which `describe_params` draws from the tool's parameters, not from a literal.
    ui = Path(controller_module.__file__).parent
    found = {
        path.name: path.read_text(encoding="utf-8").count('"confirm_turns=true"')
        for path in sorted(ui.glob("*.py"))
    }

    assert found == {
        "__init__.py": 0,
        "controller.py": 1,
        "frames.py": 0,
        "guide.py": 0,
        "plate_page.py": 0,
        "selection.py": 0,
    }


# --- the plan and the write (D7) ----------------------------------------------------

DEMO_PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "demo.toml"
#: What an earlier write left: the profile's own block, between the markers smc writes.
HELD = f"""
{BLOCK_BEGIN}
[objectives]
reference = "{TEN}"

[objectives.offsets."{FORTY}"]
xy_offset_um = [1.0, 2.0]
measured = "2026-11-01, session 2"
{BLOCK_END}
"""


@pytest.fixture
def stand_file(tmp_path: Path) -> Path:
    """A copy of ``demo.toml`` in the test's own folder: the one file a write may replace."""
    path = tmp_path / "stand.toml"
    path.write_bytes(DEMO_PROFILE.read_bytes())
    return path


@pytest.fixture
def make_stand(
    fake_core: FakeCore, stand_file: Path
) -> Iterator[Callable[[str], Microscope]]:
    """Builds the fake stand on the profile loaded from ``stand_file`` (with ``extra`` TOML appended first), so that the page's profile has a ``source``."""
    stands: list[Microscope] = []

    def make(extra: str = "") -> Microscope:
        if extra:
            stand_file.write_bytes(stand_file.read_bytes() + extra.encode("utf-8"))
        stand = Microscope.from_core(
            cast("CMMCorePlus", fake_core), Profile.load(stand_file)
        )
        stands.append(stand)
        return stand

    try:
        yield make
    finally:
        for stand in stands:
            stand.close()


def _plan_page(
    make_stand: Callable[[str], Microscope], run_doc: Path, *, held: bool
) -> PlatePage:
    """A page whose document holds the plate and a result of the 40X measured from the 10X.

    With ``held`` the profile already names the 10X as its reference and an offset for the
    40X, and the result was measured from that reference (``reference_in_profile``); the page
    holds the profile as it was when it was built.
    """
    stand = make_stand(HELD if held else "")
    _seed_plate(run_doc)
    _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25)), reference_in_profile=held))
    return _stub_page(stand, run_doc)


def _lines(was: str | None) -> tuple[str, ...]:
    """The preview of the 40X's result, written out from the numbers; ``was`` is what the profile held for the 40X (``None``: no reference at all)."""
    shift = (
        f"'{FORTY}'  shift from '{TEN}' (-37.50, +12.25) µm, spread 0.03 µm over A1, "
        "H12, A12 (bound 2.00 µm)"
    )
    if was is None:
        return (
            shift,
            f"'{FORTY}'  xy (-37.50, +12.25) µm  was unknown",
            f"reference '{TEN}'  (new: the profile named none)",
            'measured = "2026-11-14, objective-offsets in run.json"',
        )
    return (
        shift,
        f"'{FORTY}'  xy (-37.50, +12.25) µm  was xy ({was}) µm, measured "
        "2026-11-01, session 2",
        f"reference '{TEN}'",
        'measured = "2026-11-14, objective-offsets in run.json"',
    )


def test_the_plan_is_drawn_from_a_fresh_load_of_the_profile_file(
    make_stand: Callable[[str], Microscope], run_doc: Path, stand_file: Path
) -> None:
    page = _plan_page(make_stand, run_doc, held=True)
    try:
        first = page.offsets_plan("")
        # The operator edits the profile in an editor while the page is open: the page's own
        # profile, loaded when it was built, still holds (1.0, 2.0); the file holds (3.0, 4.0).
        stand_file.write_text(
            stand_file.read_text(encoding="utf-8").replace(
                "xy_offset_um = [1.0, 2.0]", "xy_offset_um = [3.0, 4.0]"
            ),
            encoding="utf-8",
        )
        second = page.offsets_plan("")
        in_memory = page._profile.objectives.offsets[FORTY].xy_offset_um
    finally:
        page.close()

    assert in_memory == (1.0, 2.0)
    assert first == OffsetsPlanView("stand.toml", _lines("+1.00, +2.00"), None, False)
    assert second == OffsetsPlanView("stand.toml", _lines("+3.00, +4.00"), None, False)


def test_the_plan_names_the_session_and_says_when_the_profile_names_no_reference(
    make_stand: Callable[[str], Microscope], run_doc: Path
) -> None:
    page = _plan_page(make_stand, run_doc, held=False)
    try:
        plain = page.offsets_plan("")
        typed = page.offsets_plan("  session 3 (#412)  ")
        bad = page.offsets_plan("two\nlines")
    finally:
        page.close()

    assert plain.lines == _lines(None)
    assert typed.lines[-1] == (
        'measured = "2026-11-14, session 3 (#412), objective-offsets in run.json"'
    )
    assert bad == OffsetsPlanView(
        "stand.toml",
        (),
        "the session is one line of text: no tab, no line break",
        False,
    )


def test_there_is_no_preview_without_a_usable_result_and_a_broken_file_is_text(
    make_stand: Callable[[str], Microscope], run_doc: Path, stand_file: Path
) -> None:
    _seed_plate(run_doc)
    page = _stub_page(make_stand(""), run_doc)
    try:
        # No record of the tool at all, then one that failed: nothing to preview, no problem.
        nothing = page.offsets_plan("")
        _seed_offsets(run_doc, status=StageStatus.failed, error="PluginRunError: no")
        failed = page.offsets_plan("")
        # A result, and a profile file that is no longer TOML: the reason, as text.
        _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25))), minute=2)
        stand_file.write_text("[microscope\nname = ", encoding="utf-8")
        broken = page.offsets_plan("")
        # A run document that cannot be read is not an empty one (FM-229).
        stand_file.write_bytes(DEMO_PROFILE.read_bytes())
        run_doc.write_text("{not json", encoding="utf-8")
        unreadable = page.offsets_plan("")
    finally:
        page.close()

    assert nothing == OffsetsPlanView("stand.toml", (), None, False)
    assert failed == OffsetsPlanView("stand.toml", (), None, False)
    assert broken.lines == ()
    assert broken.problem is not None
    assert broken.problem.startswith(str(stand_file.resolve()))
    assert unreadable.lines == ()
    assert unreadable.problem is not None
    assert unreadable.problem.startswith("run document: ")


def test_write_offsets_writes_only_the_lines_that_were_shown(
    make_stand: Callable[[str], Microscope], run_doc: Path, stand_file: Path
) -> None:
    page = _plan_page(make_stand, run_doc, held=True)
    changed = (
        "Objective offsets: the offsets or the profile changed since they were shown: "
        "look at them again (Refresh)"
    )
    try:
        shown = page.offsets_plan("session 3").lines
        before = stand_file.read_bytes()
        # (a) the session text changed since the preview: its line is not the one shown.
        page.write_offsets("session 4", shown)
        after_session = (stand_file.read_bytes(), page.status().outcome)
        # (b) the profile was edited since the preview: what is replaced is not what was shown.
        stand_file.write_text(
            stand_file.read_text(encoding="utf-8").replace(
                "xy_offset_um = [1.0, 2.0]", "xy_offset_um = [3.0, 4.0]"
            ),
            encoding="utf-8",
        )
        edited = stand_file.read_bytes()
        page.write_offsets("session 3", shown)
        after_edit = (stand_file.read_bytes(), page.status().outcome)
        # (c) the lines shown are the ones now: it writes.
        fresh = page.offsets_plan("session 3").lines
        page.write_offsets("session 3", fresh)
        written = stand_file.read_bytes()
        outcome = page.status().outcome
    finally:
        page.close()

    assert after_session == (before, changed)
    assert after_edit == (edited, changed)
    assert written != edited
    assert outcome == "Offsets written into stand.toml"
    offsets = Profile.load(stand_file).objectives.offsets[FORTY]
    assert offsets.xy_offset_um == (-37.5, 12.25)
    assert offsets.measured == "2026-11-14, session 3, objective-offsets in run.json"


def test_write_offsets_refuses_while_a_run_goes(
    make_stand: Callable[[str], Microscope], run_doc: Path, stand_file: Path
) -> None:
    page = _plan_page(make_stand, run_doc, held=False)
    try:
        shown = page.offsets_plan("").lines
        before = stand_file.read_bytes()
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        with pytest.raises(RuntimeError, match=r"^slow is running$"):
            page.write_offsets("", shown)
        after = stand_file.read_bytes()
    finally:
        Slow.release.set()
        assert page.join(10.0)
        page.close()

    assert after == before


def test_a_profile_with_no_file_has_nothing_to_write(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    assert fake_microscope.profile.source is None
    _seed_plate(run_doc)
    _seed_offsets(run_doc, _result((FORTY, (-37.5, 12.25))))
    page = _stub_page(fake_microscope, run_doc)
    try:
        view = page.offsets_plan("")
        page.write_offsets("", ())
        outcome = page.status().outcome
    finally:
        page.close()

    assert view == OffsetsPlanView(
        "",
        (),
        "this stand's profile is built into smc, with no file to write",
        False,
    )
    assert outcome == (
        "Objective offsets: this stand's profile is built into smc, with no file to write"
    )


def test_after_a_write_the_plan_says_written_and_a_second_write_is_refused_by_the_reference_check(
    make_stand: Callable[[str], Microscope], run_doc: Path, stand_file: Path
) -> None:
    page = _plan_page(make_stand, run_doc, held=False)
    refusal = (
        f"the profile's offsets are now from {TEN}, and the run measured them from "
        "(none): run objective-offsets again"
    )
    try:
        shown = page.offsets_plan("").lines
        assert page.offsets_plan("").written is False
        page.write_offsets("", shown)
        once = stand_file.read_bytes()
        after = page.offsets_plan("")
        # A second click on the same lines: the file holds a reference now, and the result
        # measured the offsets from none: the library refuses, nothing is written.
        page.write_offsets("", shown)
        twice = stand_file.read_bytes()
        outcome = page.status().outcome
        # The tool runs again: its result is not the one written, so the block says nothing
        # of a write any more.
        page.run(
            "objective-offsets",
            [f'targets=["{FORTY}"]', 'calibrations=["cal.json"]', "confirm_turns=true"],
        )
        assert page.join(10.0)
        again = page.offsets_plan("")
    finally:
        page.close()

    assert after == OffsetsPlanView("stand.toml", (), refusal, True)
    assert twice == once
    assert outcome == f"Objective offsets: {refusal}"
    assert again.written is False
    # The page's stand still holds the profile it started with, so the tool measures from the
    # old reference and the library refuses every new result: the page says to restart.
    assert again.problem == f"{refusal}; {OFFSETS_RESTART_FIRST}"


# --- what the review of #382 found ----------------------------------------------------


@pytest.mark.parametrize("objective", [None, "", "  "], ids=["none", "empty", "blank"])
def test_a_calibration_that_names_no_objective_names_nothing_to_turn_to(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    objective: str | None,
) -> None:
    # A camera calibration made at an empty turret position holds a blank label (FM-125):
    # it would be a target the confirmation asks to turn to, and the tool would refuse it.
    _seed_plate(run_doc)
    path = tmp_path / "cal-blank.json"
    seed_camera_calibration(path, objective=objective)
    page = _stub_page(fake_microscope, run_doc)
    try:
        step = page.offsets_step(_typed(path))
    finally:
        page.close()

    assert step.wait == offsets_no_objective(str(path))
    assert (step.pairs, step.targets) == ((), ())
    assert step.reads == (CalibrationRead(str(path), None, "", step.wait),)


def test_offsets_step_says_the_turret_could_not_be_read_and_guesses_nothing(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_plate(run_doc)
    forty = _document(tmp_path, FORTY, seed_camera_calibration)
    page = _stub_page(fake_microscope, run_doc)
    try:
        monkeypatch.setattr(page, "objective_labels", lambda: ())
        step = page.offsets_step(_typed(forty))
    finally:
        page.close()

    assert step.wait == OFFSETS_TURRET_UNREAD
    assert step.pairs == ()


def test_a_path_with_a_null_byte_is_a_text_for_its_line_and_never_an_exception(
    fake_microscope: Microscope, run_doc: Path
) -> None:
    _seed_plate(run_doc)
    page = _stub_page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("a\x00b.json")
    finally:
        page.close()

    (read,) = step.reads
    assert read.objective is None
    assert read.problem is not None
    assert read.problem.startswith("a\x00b.json: ValueError: ")
    assert step.wait == read.problem


def test_a_relative_path_comes_back_as_typed_in_the_line_and_in_the_pair(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read from where `smc ui` was started, as the tool reads it, and never resolved: the
    # run record would carry the operator's folder (FM-24, FM-50).
    _seed_plate(run_doc)
    seed_camera_calibration(tmp_path / "cal-40x.json", objective=FORTY)
    monkeypatch.chdir(tmp_path)
    page = _stub_page(fake_microscope, run_doc)
    try:
        step = page.offsets_step("cal-40x.json")
    finally:
        page.close()

    assert step.wait is None
    assert step.reads[0].path == "cal-40x.json"
    assert step.pairs[1] == 'calibrations=["cal-40x.json"]'


def test_offsets_step_opens_no_file_while_an_activity_goes(
    fake_microscope: Microscope,
    run_doc: Path,
    tmp_path: Path,
    seed_camera_calibration: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A path on a share that does not answer would hold the loop that handles STOP while the
    # stage moves (FM-62, FM-148): nothing is opened under a run, and the wait is the run's.
    _seed_plate(run_doc)
    forty = _document(tmp_path, FORTY, seed_camera_calibration)
    opened: list[Path] = []
    real = controller_module.calibration_from

    def spy(stage: str, path: Path, **kwargs: Any) -> Any:
        opened.append(path)
        return real(stage, path, **kwargs)

    monkeypatch.setattr(controller_module, "calibration_from", spy)
    page = _stub_page(fake_microscope, run_doc)
    try:
        page.run("slow", [])
        assert Slow.started.wait(10.0)
        during = page.offsets_step(_typed(forty))
        opened_during = list(opened)
        Slow.release.set()
        assert page.join(10.0)
        after = page.offsets_step(_typed(forty))
    finally:
        Slow.release.set()
        page.close()

    assert opened_during == []
    assert during.reads == ()
    assert during.wait == busy_note("slow", live=False)
    assert opened == [forty]
    assert after.wait is None


def test_a_write_that_fails_is_an_outcome_and_is_not_kept_as_written(
    make_stand: Callable[[str], Microscope],
    run_doc: Path,
    stand_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smc.run.files import AtomicWriteError

    page = _plan_page(make_stand, run_doc, held=False)
    try:
        shown = page.offsets_plan("").lines
        before = stand_file.read_bytes()

        def refuse(path: Path, data: bytes, *, make_parents: bool = True) -> None:
            raise AtomicWriteError(PermissionError(13, "Access is denied"), None)

        monkeypatch.setattr("smc.run.files.write_atomically", refuse)
        page.write_offsets("", shown)
        outcome = page.status().outcome
        kept = page.offsets_plan("").written
        monkeypatch.undo()
        # The file is as it was and the click may be tried again.
        unchanged = stand_file.read_bytes() == before
        page.write_offsets("", shown)
        outcome_again = page.status().outcome
    finally:
        page.close()

    assert outcome == (
        f"Objective offsets: {stand_file.resolve()} was not changed: Access is denied"
    )
    assert kept is False
    assert unchanged is True
    assert outcome_again == "Offsets written into stand.toml"
