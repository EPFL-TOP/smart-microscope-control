"""A selection made at the 10X, scanned at the 40X on its chosen fields only (m3 §28, #637).

The stand is ``plate_check_stand``'s: the camera turned 30° and mirrored, the
plate 1.2°, and the profile's offsets between objectives, so that the shift
from the 10X to the 40X is ``(-37.5, 12.25)`` µm, anisotropic (FM-73, FM-203).
The survey is made once per module through ``run_plugin`` (the seam, FM-124):
the registration, a 10X ``scan-plan`` of B2 and B3, ``scan``, ``cells
detector=blobs`` and ``route`` at the 40X, three fields a well, then the run
document copied to ``run-10x.json`` as the docs page says. Each test copies the
survey's documents into its own folder, turns the 40X in, and plans with
``fields_from=run-10x.json``; what it expects is read from the copy's raw JSON,
never from the plan it checks.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from plate_check_stand import (
    FORTY,
    ROUGH_UM,
    TEN,
    WELLS,
    build_documents,
    open_stand,
    opened,
    profile,
    truth,
)
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret, XYStage
from smc.hardware.profile import Profile
from smc.plugins import ParamsError, PluginRunError, run_plugin
from smc.plugins.cells.plugin import CellsParams, CellsPlugin
from smc.plugins.plate_walls.plugin import PlateWallsParams, PlateWallsPlugin
from smc.plugins.route.plugin import RouteParams, RoutePlugin
from smc.plugins.scan.plugin import ScanParams, ScanPlugin
from smc.plugins.scan_plan.plugin import ScanPlanParams, ScanPlanPlugin
from smc.plugins.slots import ScanPlanResult, ScanResult, WellRef
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

pytestmark = [pytest.mark.demo, pytest.mark.filterwarnings("error::RuntimeWarning")]

#: 10X -> 40X in the profile's [objectives]: (-32.5 - 5.0, 9.75 + 2.5).
SHIFT = (-37.5, 12.25)
#: How far a readback may be from its target: the demo stage lands to 0.01 µm, and
#: the smaller component of the shift is 12 times this (FM-242).
LANDING_UM = 1.0
SELECTION = Path("run-10x.json")


@dataclass(frozen=True)
class Survey:
    """The survey's documents: the run document after ``route``, its copy, the 40X's calibration."""

    run: Path
    copy: Path
    forty: Path


@pytest.fixture(scope="module")
def survey(mm_available: bool, tmp_path_factory: pytest.TempPathFactory) -> Survey:
    """The 10X survey and its route at the 40X, through ``run_plugin``; ``run-10x.json`` its copy."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    folder = tmp_path_factory.mktemp("scan-plan-selection")
    docs = build_documents(folder)
    with opened(profile()) as stand:
        assert stand.require(ObjectiveTurret).current() == TEN
        for plugin, params in (
            (ScanPlanPlugin(), ScanPlanParams(wells="B2,B3", rows=5, columns=5)),
            (ScanPlugin(), ScanParams(check_plate=False)),
            (CellsPlugin(), CellsParams(detector="blobs", model_path=None)),
            (
                RoutePlugin(),
                RouteParams(
                    objective=FORTY, fields_per_well=3, labels=("single", "doublet")
                ),
            ),
        ):
            record = run_plugin(plugin, stand, params, docs.run)
            assert record.status is StageStatus.done, record.error
    copy = folder / SELECTION.name
    shutil.copyfile(docs.run, copy)
    return Survey(docs.run, copy, docs.forty)


@pytest.fixture
def stands(mm_available: bool) -> Iterator[list[Microscope]]:
    """The stands a test opened, closed afterwards."""
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    opened_: list[Microscope] = []
    yield opened_
    for stand in opened_:
        stand.close()


@pytest.fixture
def here(survey: Survey, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The survey's documents copied into the test's folder, which is the working directory."""
    for source in (survey.run, survey.copy, survey.forty):
        shutil.copyfile(source, tmp_path / source.name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _stand(
    stands: list[Microscope], the_profile: Profile | None = None, objective: str = FORTY
) -> Microscope:
    stand = open_stand(the_profile if the_profile is not None else profile())
    stands.append(stand)
    turret = stand.require(ObjectiveTurret)
    if turret.current() != objective:
        turret.select(objective, confirm=True)
    return stand


def _plan(stand: Microscope, **params: Any) -> StageRecord:
    given: dict[str, Any] = {
        "wells": "all",
        "fields_from": SELECTION,
        "calibration": Path("cal-40x.json"),
        **params,
    }
    return run_plugin(
        ScanPlanPlugin(), stand, ScanPlanParams(**given), Path("run.json")
    )


def _planned(record: StageRecord) -> ScanPlanResult:
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    return ScanPlanResult.model_validate(record.result)


def _refused(stand: Microscope, **params: Any) -> str:
    """Plan and expect a refusal of ``scan-plan``, raised outside any handler (FM-98)."""
    with pytest.raises(PluginRunError) as raised:
        _plan(stand, **params)
    assert raised.value.plugin == "scan-plan"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    stored = RunDocument.load(Path("run.json")).stages["scan-plan"]
    assert stored.status is StageStatus.failed
    assert stored.result is None
    return raised.value.reason


def _raw(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _stops(path: Path) -> list[dict[str, Any]]:
    """The route's stops of a document's raw JSON, in the route's order."""
    positions: list[dict[str, Any]] = _raw(path)["stages"]["route"]["result"][
        "positions"
    ]
    return sorted(positions, key=lambda stop: stop["order"])


def _rewrite_route(path: Path, **record: Any) -> dict[str, Any]:
    """The route's record of ``path`` changed by ``record`` (or its result by ``result=``), validated and written (§20.1)."""
    dumped = RunDocument.load(path).stages["route"].model_dump(mode="json")
    dumped.update(record)
    write_stage(path, StageRecord.model_validate(dumped))
    return dumped


# ------------------------------------------------------------------ the walk


def test_a_selection_at_10x_is_scanned_at_40x_on_its_chosen_fields_in_its_order(
    stands: list[Microscope], here: Path
) -> None:
    stops = _stops(SELECTION)
    assert len(stops) == 6
    stand = _stand(stands)

    plan = _planned(_plan(stand))
    record = run_plugin(ScanPlugin(), stand, ScanParams(), Path("run.json"))
    assert record.status is StageStatus.done, record.error
    assert record.result is not None
    scan = ScanResult.model_validate(record.result)

    # Row B of an "all" plan, serpentine: its second row, visited right to left, so
    # B3 before B2; each well's stops in the route's order.
    assert plan.wells == ("B3", "B2")
    expected = [stop for well in ("B3", "B2") for stop in stops if stop["well"] == well]
    assert [field.stop for field in plan.fields] == [stop["name"] for stop in expected]
    assert [frame.name for frame in scan.frames] == [f.name for f in plan.fields]
    for frame, stop in zip(scan.frames, expected, strict=True):
        assert frame.x_um == pytest.approx(stop["x_um"] - 37.5, abs=LANDING_UM)
        assert frame.y_um == pytest.approx(stop["y_um"] + 12.25, abs=LANDING_UM)

    selection = plan.selection
    assert selection is not None
    assert selection.run_doc == "run-10x.json"
    assert selection.route_attempt == _raw(SELECTION)["stages"]["route"]["attempt"]
    assert selection.selector == "most-cells"
    assert selection.source_objective == TEN
    assert selection.shift_um == SHIFT
    assert selection.fields_left_out == 0
    assert len(selection.wells_without_fields) == 94
    assert (plan.rows, plan.columns, plan.overlap) == (1, 1, 0.0)
    assert plan.objective == FORTY
    # The selection's lines come after the frame's (none: the offset is measured)
    # and before the camera's, which says nothing of a grid's gaps.
    assert plan.assumed[0] == (
        f"the fields are the 6 that selector 'most-cells' chose in run-10x.json "
        f"(measured {selection.measured}), each well's in the selection's order"
    )
    assert plan.assumed[1].startswith("no field was chosen in 94 of the wells named:")
    assert not any("overlap" in line for line in plan.assumed)
    assert scan.assumed[-1] == (
        "the plan's 6 fields in 2 wells were chosen by selector 'most-cells' in "
        "run-10x.json: the rest of each well was not scanned"
    )


# ------------------------------------------------------------------ refused


def test_a_selection_on_another_registration_of_the_plate_is_refused(
    stands: list[Microscope], here: Path
) -> None:
    old = _raw(Path("run.json"))["stages"]["plate-walls"]["attempt"]
    stand = _stand(stands, objective=TEN)
    refs = tuple(
        WellRef(
            well=well,
            x_um=truth(well)[0] + ROUGH_UM[well][0],
            y_um=truth(well)[1] + ROUGH_UM[well][1],
        )
        for well in WELLS
    )
    again = run_plugin(
        PlateWallsPlugin(),
        stand,
        PlateWallsParams(plate="96-well", refs=refs),
        Path("run.json"),
    )
    assert again.status is StageStatus.done, again.error
    stand.require(ObjectiveTurret).select(FORTY, confirm=True)

    assert _refused(stand) == (
        f"the selection in run-10x.json was chosen on another registration of the "
        f"plate ({old} against this run's {again.attempt})"
    )


def test_a_selection_of_another_objectives_fields_is_refused_naming_both(
    stands: list[Microscope], here: Path
) -> None:
    stand = _stand(stands, objective=TEN)
    assert _refused(stand, calibration=None) == (
        f"the selection in run-10x.json chose fields of {FORTY}, the plan is for "
        f"{TEN}: run route there with objective={TEN}, or plan for {FORTY}"
    )


def test_a_selection_field_outside_its_well_is_refused_by_its_stop(
    stands: list[Microscope], here: Path
) -> None:
    stops = _stops(SELECTION)
    moved = next(stop for stop in stops if stop["well"] == "B2")
    result = _raw(SELECTION)["stages"]["route"]["result"]
    for stop in result["positions"]:
        if stop["name"] == moved["name"]:
            stop["x_um"] += 5000.0
    _rewrite_route(SELECTION, result=result)
    stand = _stand(stands)

    x_um, y_um = moved["x_um"] + 5000.0 - 37.5, moved["y_um"] + 12.25
    assert _refused(stand) == (
        f"field {moved['name']} of the selection lies at ({x_um:.1f}, {y_um:.1f}) µm "
        "in the plan's frame, outside well B2: check the offsets of the profile's "
        "[objectives], or run route again"
    )


def test_a_selection_field_outside_the_soft_limits_is_refused_before_anything_moves(
    stands: list[Microscope], here: Path
) -> None:
    stops = _stops(SELECTION)
    highest = max(stops, key=lambda stop: stop["y_um"])
    y_hi = highest["y_um"] + 12.25 - 1.0
    assert all(stop["y_um"] + 12.25 <= y_hi for stop in stops if stop is not highest)
    base = profile()
    limited = base.model_copy(
        update={
            "safety": base.safety.model_copy(
                update={"xy_soft_limits_um": ((-49_000.0, 49_000.0), (-29_000.0, y_hi))}
            )
        }
    )
    stand = _stand(stands, limited)
    before = stand.require(XYStage).position_um()

    reason = _refused(stand)

    x_um, y_um = highest["x_um"] - 37.5, highest["y_um"] + 12.25
    assert reason == (
        f"field {highest['name']} of the selection lies at ({x_um:.1f}, {y_um:.1f}) "
        f"µm, outside the stage's soft limits x [-49000, 49000], y [-29000, "
        f"{y_hi:g}] µm"
    )
    assert stand.require(XYStage).position_um() == before


def test_a_selection_with_no_field_in_the_named_wells_is_refused(
    stands: list[Microscope], here: Path
) -> None:
    stand = _stand(stands)
    # The route's wells in the order first met in its order.
    theirs = ", ".join(dict.fromkeys(stop["well"] for stop in _stops(SELECTION)))
    assert _refused(stand, wells="A1,H12") == (
        f"the selection in run-10x.json holds no field in the wells named (A1,H12); "
        f"it chose fields in {theirs}"
    )


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_fields_from_naming_this_run_document_is_refused_before_anything_is_written(
    spelling: str, stands: list[Microscope], here: Path
) -> None:
    before = Path("run.json").read_bytes()
    stand = _stand(stands)
    named = Path("run.json") if spelling == "relative" else here / "run.json"

    with pytest.raises(ParamsError) as raised:
        _plan(stand, fields_from=named)

    assert str(raised.value) == (
        f"parameters of scan-plan: fields_from names the run document this run "
        f"writes ({named.as_posix()}), where this run replaces the scan-plan that "
        f"its route was made from: name a copy of it or another run document "
        f"(nothing was written). smc run scan-plan --show-params lists the keys"
    )
    assert Path("run.json").read_bytes() == before


def test_a_dry_run_route_is_refused_for_a_real_plan(
    stands: list[Microscope], here: Path
) -> None:
    _rewrite_route(SELECTION, dry_run=True)
    stand = _stand(stands)
    assert _refused(stand) == (
        "in run-10x.json: route was a dry run; run it without --dry-run"
    )


def test_an_unmeasured_offset_plans_the_selections_positions_and_says_so(
    stands: list[Microscope], here: Path
) -> None:
    stops = _stops(SELECTION)
    stand = _stand(stands, profile(forty=False))

    plan = _planned(_plan(stand))

    by_name = {stop["name"]: stop for stop in stops}
    for field in plan.fields:
        stop = by_name[field.stop or ""]
        assert (field.x_um, field.y_um) == (stop["x_um"], stop["y_um"])
    assert plan.selection is not None
    assert plan.selection.shift_um is None
    line = (
        f"the offset between {TEN} and {FORTY} is not measured (the profile's "
        f"[objectives]): the selection's positions are {TEN}'s, not corrected"
    )
    assert line in plan.assumed
    # After the selection's first line, before the wells it left out.
    assert plan.assumed.index(line) == (
        plan.assumed.index(
            next(a for a in plan.assumed if a.startswith("the fields are the 6"))
        )
        + 1
    )
