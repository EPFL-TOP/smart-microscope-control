"""``route`` chooses and orders through a selector and an orderer by name (algorithms design §5.6, §5.8, #567).

No stand is opened (FM-40): ``Microscope.offline(Profile.demo())`` runs the plugin on a
run folder that ``select_fields_seed.seed_run`` writes (the plan, a scan with a frame at
every planned field and no TIFF, the cells). The test algorithms are classes of this
module, put in the registry by replacing ``registry.entry_points`` next to the
built-ins, and they do what ``SCRIPT`` says, so that one class serves every case.
Every expected value is worked out here from ``choose_fields``, ``order_route``,
``footprint`` and hand-placed points, never from the result's own fields (FM-124), on
cameras that are turned, mirrored and not square (FM-73).
"""

from __future__ import annotations

import itertools
import math
import os
import random
import re
import sys
import threading
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from select_fields_seed import (
    CENTRES,
    PIXEL_UM,
    SHAPE,
    Seed,
    cell_at,
    extent_of,
    matrix_of,
    seed_run,
)
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins import PluginRunError, Progress, run_plugin
from smc.plugins.geometry import Matrix2
from smc.plugins.nis_export.format import decode
from smc.plugins.nis_export.plugin import NisExportParams, NisExportPlugin
from smc.plugins.route.algorithm import choose_fields, order_route
from smc.plugins.route.plugin import RouteParams, RoutePlugin
from smc.plugins.slots import Cell, NisExportResult, RouteResult, well_geometry
from smc.run import RunDocument, StageStatus, write_stage
from smc.selection import registry
from smc.selection.api import (
    Choice,
    OrdererParams,
    SelectionInput,
    SelectorParams,
)

THIS = __name__
WELLS = ("A1", "A2", "B2", "B1")  # a non-raster order: the scan's
EUCLIDEAN = (
    "the route minimises the Euclidean distance between fields; whether the "
    "stage's move time follows it is not measured"
)

#: What the test selector and the test orderer do, set by the test that needs it.
SCRIPT: dict[str, Any] = {}


class ScriptedParams(SelectorParams):
    pass


class Scripted:
    name = "scripted"
    summary = "Does what the test says."
    Params = ScriptedParams

    def select(self, data: SelectionInput, params: ScriptedParams) -> Sequence[Choice]:
        return SCRIPT["select"](data)  # type: ignore[no-any-return]


class ScriptedOrdererParams(OrdererParams):
    pass


class ScriptedOrderer:
    name = "scripted-order"
    summary = "Orders as the test says."
    Params = ScriptedOrdererParams

    def order(
        self,
        choices: Sequence[Choice],
        data: SelectionInput,
        params: ScriptedOrdererParams,
    ) -> Sequence[int]:
        return SCRIPT["order"](choices, data)  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-ins and the two scripted algorithms in place of the installed ones; nothing scripted yet."""

    def entry(name: str, module: str, target: str, group: str) -> EntryPoint:
        return EntryPoint(name, f"{module}:{target}", group)

    selectors = [
        entry(
            "most-cells",
            "smc.selection.builtin",
            "MostCells",
            registry.SELECTOR_GROUP,
        ),
        entry("scripted", THIS, "Scripted", registry.SELECTOR_GROUP),
    ]
    orderers = [
        entry("nearest", "smc.selection.builtin", "Nearest", registry.ORDERER_GROUP),
        entry("as-chosen", "smc.selection.builtin", "AsChosen", registry.ORDERER_GROUP),
        entry("scripted-order", THIS, "ScriptedOrderer", registry.ORDERER_GROUP),
    ]
    groups = {registry.SELECTOR_GROUP: selectors, registry.ORDERER_GROUP: orderers}
    monkeypatch.setattr(registry, "entry_points", lambda *, group: list(groups[group]))
    SCRIPT.clear()


@pytest.fixture
def offline() -> Microscope:
    return Microscope.offline(Profile.demo())


def _run(
    microscope: Microscope,
    tmp_path: Path,
    *,
    stop: threading.Event | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    **params: Any,
) -> Any:
    return run_plugin(
        RoutePlugin(),
        microscope,
        RouteParams(**params),
        tmp_path / "run.json",
        stop=stop,
        on_progress=on_progress,
    )


def _result(record: Any) -> RouteResult:
    assert record.status is StageStatus.done, record.error
    return RouteResult.model_validate(record.result)


def _refused(microscope: Microscope, tmp_path: Path, **params: Any) -> str:
    """Run, expect the stage to end ``failed``, and give the reason the record keeps."""
    with pytest.raises(PluginRunError) as raised:
        _run(microscope, tmp_path, **params)
    stored = RunDocument.load(tmp_path / "run.json").stages["route"]
    assert stored.status is StageStatus.failed
    assert stored.result is None
    return raised.value.reason


def _random_cells(seed: int, labels: Sequence[str]) -> list[Cell]:
    """Clusters of cells in every well, placed by a seeded generator: some are reachable by a field, some are not."""
    generator = random.Random(seed)
    cells: list[Cell] = []
    for well in WELLS:
        for _ in range(3):
            cx = generator.uniform(-2200.0, 2200.0)
            cy = generator.uniform(-2200.0, 2200.0)
            for _ in range(generator.randint(3, 9)):
                cells.append(
                    cell_at(
                        well,
                        cx + generator.uniform(-220.0, 220.0),
                        cy + generator.uniform(-220.0, 220.0),
                        generator.choice(labels),
                    )
                )
    return cells


def _seed(tmp_path: Path, cells: list[Cell], **kwargs: Any) -> Seed:
    return seed_run(
        tmp_path,
        cells=cells,
        wells=kwargs.pop("wells", WELLS),
        write_frames=False,
        **kwargs,
    )


# ------------------------------------------------------------- the default route


@pytest.mark.parametrize(
    ("seed_number", "matrix", "labels", "extra"),
    [
        pytest.param(1, matrix_of(PIXEL_UM, 0.0, False), ("single",), {}, id="upright"),
        pytest.param(
            2,
            matrix_of(PIXEL_UM, 17.0, True),
            ("single", "doublet"),
            {"fields_per_well": 4, "wall_margin_um": 100.0},
            id="turned-mirrored-two-labels",
        ),
        pytest.param(
            3,
            matrix_of(PIXEL_UM, -33.0, False),
            ("single", "doublet", "debris"),
            {"min_cells": 2, "exclude_centre_um": 300.0},
            id="turned-three-labels",
        ),
    ],
)
def test_the_default_selector_and_orderer_give_the_route_of_m3_11(
    tmp_path: Path,
    offline: Microscope,
    seed_number: int,
    matrix: Matrix2,
    labels: tuple[str, ...],
    extra: dict[str, Any],
) -> None:
    seed = _seed(tmp_path, _random_cells(seed_number, labels), matrix=matrix)
    parameters = {
        "labels": list(labels),
        "candidate_step_fraction": 0.5,
        "fields_per_well": 10,
        "min_cells": 1,
        "exclude_centre_um": 0.0,
        "wall_margin_um": 0.0,
        **extra,
    }

    result = _result(_run(offline, tmp_path, **parameters))

    # The oracle: m3 §11 as the plugin made it before the algorithms were named.
    step = parameters["candidate_step_fraction"] * min(extent_of(matrix, SHAPE))
    expected: list[tuple[str, str, float, float, dict[str, int]]] = []
    uncovered: dict[str, int] = {}
    previous: tuple[float, float] | None = None
    for well in WELLS:
        chosen = choose_fields(
            seed.cells,
            well_geometry(seed.plan.plate_plan, well),
            labels=labels,
            matrix=matrix,
            image_shape_px=SHAPE,
            n=parameters["fields_per_well"],
            min_cells=parameters["min_cells"],
            exclude_centre_um=parameters["exclude_centre_um"],
            wall_margin_um=parameters["wall_margin_um"],
            step_um=step,
        )
        wanted = sum(1 for c in seed.cells if c.well == well and c.label in set(labels))
        uncovered[well] = wanted - sum(sum(f.counts.values()) for f in chosen)
        if not chosen:
            continue
        reference = previous or next(
            (f.x_um, f.y_um) for f in seed.plan.fields if f.well == well
        )
        points = [(f.x_um, f.y_um) for f in chosen]
        # The nearest to the reference, as a scan: strictly nearer replaces, so a tie goes to the first.
        start = 0
        for i in range(1, len(points)):
            if math.dist(points[i], reference) < math.dist(points[start], reference):
                start = i
        for k, index in enumerate(order_route(points, start=start)):
            field = chosen[index]
            expected.append(
                (well, f"{well}_r{k:03d}", field.x_um, field.y_um, dict(field.counts))
            )
        previous = (expected[-1][2], expected[-1][3])
    assert expected, "the seed must give the oracle something to choose"

    assert [
        (p.order, p.well, p.name, p.x_um, p.y_um, p.counts) for p in result.positions
    ] == [(i, *stop) for i, stop in enumerate(expected)]
    assert result.uncovered == uncovered
    assert result.distance_um == pytest.approx(
        sum(
            math.dist((a[2], a[3]), (b[2], b[3]))
            for a, b in itertools.pairwise(expected)
        ),
        rel=1e-12,
    )
    # Written exactly as it always was: no score, no measures, no new line of `assumed`.
    stored = RunDocument.load(tmp_path / "run.json").stages["route"].result
    assert stored is not None
    for position in stored["positions"]:
        assert set(position) == {
            "order",
            "well",
            "name",
            "x_um",
            "y_um",
            "z_um",
            "pfs_offset",
            "counts",
        }
    assert result.assumed[0] == EUCLIDEAN
    assert not any("selector" in line or "orderer" in line for line in result.assumed)


# -------------------------------------------------- a selector registered by name

A1 = CENTRES["A1"]


def _touching_pair(order: Sequence[int]) -> Callable[[SelectionInput], list[Choice]]:
    """Two fields of A1 that touch (a width, 400 µm, apart), listed in ``order``, with a score and measures on the first."""
    pair = [
        Choice("A1", A1[0], A1[1], score=0.9, measures={"density_per_mm2": 12.5}),
        Choice("A1", A1[0] + 400.0, A1[1]),
    ]
    return lambda data: [pair[i] for i in order]


def test_a_registered_selector_runs_through_route_by_name_with_its_score_and_measures(
    tmp_path: Path, offline: Microscope
) -> None:
    cells = [
        cell_at("A1", 200.0, 0.0, "single"),  # on the edge the two fields share
        cell_at("A1", -50.0, 10.0, "single"),  # held by the first only
        cell_at("A1", 450.0, -20.0, "doublet"),  # held by the second only
    ]
    _seed(tmp_path, cells, wells=("A1",))
    SCRIPT["select"] = _touching_pair([0, 1])

    result = _result(
        _run(offline, tmp_path, selector="scripted", labels=["single", "doublet"])
    )

    first, second = sorted(result.positions, key=lambda p: p.x_um)
    assert (first.x_um, first.score, first.measures) == (
        A1[0],
        0.9,
        {"density_per_mm2": 12.5},
    )
    assert (second.x_um, second.score, second.measures) == (A1[0] + 400.0, None, {})
    # The cell on the shared edge goes to the first of the selector's choices.
    assert first.counts == {"single": 2, "doublet": 0}
    assert second.counts == {"single": 0, "doublet": 1}
    assert result.uncovered == {"A1": 0}
    # The stored position says what the selector gave and leaves out what it did not.
    stored = RunDocument.load(tmp_path / "run.json").stages["route"].result
    assert stored is not None
    by_x = {p["x_um"]: p for p in stored["positions"]}
    assert by_x[A1[0]]["score"] == 0.9
    assert by_x[A1[0]]["measures"] == {"density_per_mm2": 12.5}
    assert "score" not in by_x[A1[0] + 400.0]
    assert "measures" not in by_x[A1[0] + 400.0]
    # The result says whose choice it is, after what it always said.
    assert result.assumed[0] == EUCLIDEAN
    assert result.assumed[-1] == (
        "the fields were chosen by selector 'scripted', not by route's own rule of "
        "most cells: its choice, its score and its measures are its own"
    )

    # The other way round, the shared cell goes to the other field.
    SCRIPT["select"] = _touching_pair([1, 0])
    swapped = _result(
        _run(offline, tmp_path, selector="scripted", labels=["single", "doublet"])
    )
    first, second = sorted(swapped.positions, key=lambda p: p.x_um)
    assert first.counts == {"single": 1, "doublet": 0}
    assert second.counts == {"single": 1, "doublet": 1}


def test_a_registered_orderer_orders_the_whole_plate_across_wells(
    tmp_path: Path, offline: Microscope
) -> None:
    a2 = CENTRES["A2"]
    layout = [("A1", A1, 0.0), ("A1", A1, 600.0), ("A2", a2, 0.0), ("A2", a2, 600.0)]
    cells = [cell_at(well, dx, 0.0, "single") for well, _, dx in layout]
    cells += [cell_at("A1", 2000.0, 0.0), cell_at("A2", 2000.0, 0.0)]  # held by none
    _seed(tmp_path, cells, wells=("A1", "A2"))
    SCRIPT["select"] = lambda data: [Choice(w, c[0] + dx, c[1]) for w, c, dx in layout]
    SCRIPT["order"] = lambda choices, data: list(reversed(range(len(choices))))
    seen: list[Progress] = []

    result = _result(
        _run(
            offline,
            tmp_path,
            selector="scripted",
            orderer="scripted-order",
            on_progress=seen.append,
        )
    )

    # The whole plate in the orderer's order: A2's stops, then A1's, the last choice first.
    assert [(p.order, p.well, p.name, p.x_um) for p in result.positions] == [
        (0, "A2", "A2_r000", a2[0] + 600.0),
        (1, "A2", "A2_r001", a2[0]),
        (2, "A1", "A1_r000", A1[0] + 600.0),
        (3, "A1", "A1_r001", A1[0]),
    ]
    points = [(p.x_um, p.y_um) for p in result.positions]
    assert result.distance_um == pytest.approx(
        sum(math.dist(a, b) for a, b in itertools.pairwise(points)), rel=1e-12
    )
    assert result.uncovered == {"A1": 1, "A2": 1}
    assert [p.counts for p in result.positions] == [{"single": 1}] * 4
    # Not route's shortest path, and the line says whose order it is.
    assert EUCLIDEAN not in result.assumed
    assert result.assumed[0] == (
        "the visiting order is orderer 'scripted-order''s, not route's shortest "
        "path: distance_um is the length of that order's path, and whether the "
        "stage's move time follows it is not measured"
    )
    # The progress is still per well, in the plan's order.
    assert [(p.done, p.total, p.message) for p in seen] == [(1, 2, "A1"), (2, 2, "A2")]


def test_the_selector_is_given_what_route_read(
    tmp_path: Path, offline: Microscope
) -> None:
    matrix = matrix_of(PIXEL_UM, 17.0, True)
    cells = _random_cells(7, ("single", "doublet"))
    seed = _seed(tmp_path, cells, matrix=matrix)
    given: list[SelectionInput] = []

    def record(data: SelectionInput) -> list[Choice]:
        given.append(data)
        return []

    SCRIPT["select"] = record

    _refused(
        offline,
        tmp_path,
        selector="scripted",
        labels=["single", "doublet"],
        fields_per_well=3,
        wall_margin_um=120.0,
        exclude_centre_um=50.0,
    )

    (data,) = given
    assert data.scan.wells == WELLS
    assert list(data.scan.cells or ()) == list(seed.cells)
    assert set(data.scan.document.stages) >= {
        "plate-registration",
        "scan-plan",
        "scan",
        "cells",
    }
    assert data.scan.document.stages["scan"].result == seed_scan_dump(tmp_path)
    assert data.labels == ("single", "doublet")
    assert (data.fields_per_well, data.wall_margin_um, data.exclude_centre_um) == (
        3,
        120.0,
        50.0,
    )
    assert data.objective == seed.scan.objective
    assert data.matrix_um_per_px == matrix
    assert data.image_shape_px == SHAPE
    assert data.scan.run_dir == tmp_path
    assert data.first_field("B1") == next(
        (f.x_um, f.y_um) for f in seed.plan.fields if f.well == "B1"
    )


def seed_scan_dump(tmp_path: Path) -> dict[str, Any]:
    """The ``scan`` record's result as the run folder holds it."""
    stored = RunDocument.load(tmp_path / "run.json").stages["scan"].result
    assert stored is not None
    return stored


# ------------------------------------------------------------ what the layer refuses

A1_CENTRE = Choice("A1", A1[0], A1[1])


def _far(dx: float) -> Choice:
    return Choice("A1", A1[0] + dx, A1[1])


@pytest.mark.parametrize(
    ("choices", "extra", "reason"),
    [
        (
            lambda: [A1_CENTRE, Choice("C3", A1[0], A1[1])],
            {},
            r"choice 1 \(C3 at x=.*\): its well is not a well of this scan \(A1, A2\)",
        ),
        (
            lambda: [Choice("A1", math.nan, A1[1])],
            {},
            r"choice 0 \(A1 at x=nan.*\): x_um is not finite \(nan\)",
        ),
        (
            lambda: [_far(3100.0)],
            {},
            r"choice 0 \(A1 at x=.*\): the field leaves the well shrunk by "
            r"wall_margin_um=0 µm or touches the disc of exclude_centre_um=0 µm",
        ),
        (
            lambda: [_far(0.0), _far(500.0), _far(1000.0)],
            {"fields_per_well": 2},
            r"well A1 has 3 choices, more than fields_per_well=2",
        ),
        (
            lambda: [_far(0.0), _far(100.0)],
            {},
            r"choice 0 \(A1 at x=.*\): it overlaps choice 1: two fields of one "
            r"well must not overlap \(touching is allowed\)",
        ),
        (
            lambda: [Choice("A1", A1[0], A1[1], measures={"Bad Key": 1.0})],
            {},
            r"choice 0 \(A1 at x=.*\): measure key 'Bad Key' is not snake_case \(.*\)",
        ),
        (
            lambda: [A1_CENTRE, ("A1", 0.0, 0.0)],
            {},
            r"item 1 is a tuple, not a Choice",
        ),
    ],
    ids=[
        "a-well-of-no-scan",
        "a-nan",
        "outside-the-well",
        "too-many",
        "overlap",
        "a-bad-measure-key",
        "not-a-choice",
    ],
)
def test_each_check_ends_the_stage_failed_and_names_the_selector_and_the_choice(
    tmp_path: Path,
    offline: Microscope,
    choices: Callable[[], list[Any]],
    extra: dict[str, Any],
    reason: str,
) -> None:
    # Two wells, so that three choices in one are more than its limit and not yet more than the plate's.
    _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1", "A2"))
    SCRIPT["select"] = lambda data: choices()

    text = _refused(offline, tmp_path, selector="scripted", **extra)

    assert re.fullmatch(rf"selector scripted: {reason}", text), text


def test_a_repeated_index_in_the_order_ends_the_stage_failed_and_names_the_orderer(
    tmp_path: Path, offline: Microscope
) -> None:
    _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1",))
    SCRIPT["select"] = lambda data: [_far(0.0), _far(600.0)]
    SCRIPT["order"] = lambda choices, data: [0, 0]

    text = _refused(offline, tmp_path, selector="scripted", orderer="scripted-order")

    assert text == (
        "orderer scripted-order: the order is not a permutation of 0..1: repeats index 0"
    )


def test_a_choice_in_a_well_the_cells_stage_did_not_scan_is_refused_and_not_dropped(
    tmp_path: Path, offline: Microscope
) -> None:
    # A2 is in the plan, with its fields, and the cells stage found nothing to scan there.
    seed = _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1", "A2"))
    assert set(seed.result.counts) == {"A1", "A2"}
    document = RunDocument.load(tmp_path / "run.json")
    cells_record = document.stages["cells"]
    result = dict(cells_record.result or {})
    counts = dict(result["counts"])
    del counts["A2"]
    result["counts"] = counts
    write_stage(
        tmp_path / "run.json", cells_record.model_copy(update={"result": result})
    )
    a2 = CENTRES["A2"]
    SCRIPT["select"] = lambda data: [Choice("A2", a2[0], a2[1])]

    text = _refused(offline, tmp_path, selector="scripted")

    assert text == (
        "selector scripted: choice 0 is in well A2, which the cells stage did not "
        "scan (A1); a route covers the wells that have cells"
    )


# ------------------------------------------------------------ what an algorithm breaks


def _boom(data: SelectionInput) -> list[Choice]:
    raise KeyError("missing-key")


def _late(data: SelectionInput) -> Any:
    yield A1_CENTRE
    raise ValueError("late: raised while the list was read")


def test_an_exception_in_a_selector_or_an_orderer_ends_the_stage_failed_with_its_name_its_type_and_where(
    tmp_path: Path, offline: Microscope
) -> None:
    _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1",))
    where = r"\(at test_route_selection\.py:\d+\)"

    SCRIPT["select"] = _boom
    text = _refused(offline, tmp_path, selector="scripted")
    assert re.fullmatch(rf"selector scripted: KeyError: 'missing-key' {where}", text)
    # The file's base name and a line: never the path of whoever is logged on (FM-50).
    assert str(Path(__file__).parent) not in text
    assert "/" not in text.split("(at", 1)[1]

    # A generator that raises while `route` reads the list is the same refusal.
    SCRIPT["select"] = _late
    text = _refused(offline, tmp_path, selector="scripted")
    assert re.fullmatch(
        rf"selector scripted: ValueError: late: raised while the list was read {where}",
        text,
    )

    # None is a refusal in words, not "NoneType is not iterable".
    SCRIPT["select"] = lambda data: None
    assert _refused(offline, tmp_path, selector="scripted") == (
        "selector scripted: select returned None, not a list"
    )

    # The orderer likewise.
    SCRIPT["select"] = lambda data: [A1_CENTRE]

    def broken_order(choices: Sequence[Choice], data: SelectionInput) -> list[int]:
        raise ArithmeticError("the order could not be worked out")

    SCRIPT["order"] = broken_order
    text = _refused(offline, tmp_path, selector="scripted", orderer="scripted-order")
    assert re.fullmatch(
        rf"orderer scripted-order: ArithmeticError: the order could not be worked "
        rf"out {where}",
        text,
    )
    SCRIPT["order"] = lambda choices, data: None
    assert _refused(
        offline, tmp_path, selector="scripted", orderer="scripted-order"
    ) == ("orderer scripted-order: order returned None, not a list")


def test_the_refusal_keeps_no_traceback_of_the_algorithm(
    tmp_path: Path, offline: Microscope
) -> None:
    # FM-98: the sentence is built inside the handler and raised outside it, so the
    # refusal holds no cause and no context, and with them none of the algorithm's frames.
    _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1",))
    SCRIPT["select"] = _boom

    with pytest.raises(PluginRunError) as raised:
        _run(offline, tmp_path, selector="scripted")

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None

    SCRIPT["select"] = lambda data: [
        ("A1", 0.0, 0.0)
    ]  # a check refuses, in the same helper
    with pytest.raises(PluginRunError) as checked:
        _run(offline, tmp_path, selector="scripted")

    assert checked.value.__cause__ is None
    assert checked.value.__context__ is None


def test_where_an_exception_was_raised_is_found_when_the_modules_file_differs_in_case(
    tmp_path: Path, offline: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows a module's `__file__` and a frame's file name may differ in case
    # (FM-43): `os.path.normcase` is the identity here, so it is replaced by a
    # function that folds the case, and the module's file is spelt another way.
    _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1",))
    SCRIPT["select"] = _boom
    module = sys.modules[THIS]
    monkeypatch.setattr(os.path, "normcase", str.lower)
    monkeypatch.setattr(module, "__file__", str(module.__file__).upper())

    text = _refused(offline, tmp_path, selector="scripted")

    assert re.search(r"\(at test_route_selection\.py:\d+\)$", text), text


def test_a_keyboard_interrupt_in_a_selector_is_not_a_refusal(
    tmp_path: Path, offline: Microscope
) -> None:
    _seed(tmp_path, [cell_at("A1", 0.0, 0.0)], wells=("A1",))

    def interrupted(data: SelectionInput) -> list[Choice]:
        raise KeyboardInterrupt

    SCRIPT["select"] = interrupted

    with pytest.raises(KeyboardInterrupt):
        _run(offline, tmp_path, selector="scripted")


# ------------------------------------------------------------- stop and progress


def test_a_stop_and_the_progress_stay_per_well_with_a_selector(
    tmp_path: Path, offline: Microscope
) -> None:
    _seed(
        tmp_path, _random_cells(11, ("single",)), matrix=matrix_of(PIXEL_UM, 17.0, True)
    )
    seen: list[Progress] = []
    at_the_selector: list[int] = []

    def choose(data: SelectionInput) -> list[Choice]:
        at_the_selector.append(len(seen))
        return [Choice(w, *CENTRES[w]) for w in data.scan.wells]

    SCRIPT["select"] = choose

    _result(_run(offline, tmp_path, selector="scripted", on_progress=seen.append))

    # The selector is one step for every well, before any progress; then one per well.
    assert at_the_selector == [0]
    assert [(p.done, p.total, p.message) for p in seen] == [
        (1, 4, "A1"),
        (2, 4, "A2"),
        (3, 4, "B2"),
        (4, 4, "B1"),
    ]

    # A stop set at the first progress ends the stage `stopped`, between two wells.
    seen.clear()
    stop = threading.Event()

    def stopping(progress: Progress) -> None:
        seen.append(progress)
        stop.set()

    record = _run(
        offline, tmp_path, selector="scripted", stop=stop, on_progress=stopping
    )
    assert record.status is StageStatus.stopped
    assert record.result is None
    assert [(p.done, p.total, p.message) for p in seen] == [(1, 4, "A1")]


def test_a_stop_before_the_selector_never_calls_it(
    tmp_path: Path, offline: Microscope
) -> None:
    _seed(tmp_path, _random_cells(11, ("single",)))
    calls: list[int] = []
    SCRIPT["select"] = lambda data: calls.append(1) or []
    stop = threading.Event()
    stop.set()

    record = _run(offline, tmp_path, selector="scripted", stop=stop)

    assert record.status is StageStatus.stopped
    assert calls == []


# ------------------------------------------------------------------ no route


def test_no_stop_at_all_is_the_old_refusal_for_most_cells_and_a_selector_refusal_for_another(
    tmp_path: Path, offline: Microscope
) -> None:
    cells = [cell_at("A1", 0.0, 0.0, "single")]
    _seed(tmp_path, cells, wells=("A1",))

    # most-cells: today's two texts.
    assert _refused(offline, tmp_path, labels=["doublet"]) == (
        "the cells stage found no doublet cell: there is nothing to route; "
        "choose other labels, or check the scan and the cells"
    )
    assert _refused(offline, tmp_path, min_cells=5) == (
        "no field of 400 by 300 µm holds 5 or more single cell and keeps to the "
        "wells' margins; lower min_cells, wall_margin_um or exclude_centre_um, or "
        "check the scan's pixel size"
    )

    # Another selector that chooses nothing, and the orderer it is never asked for.
    asked: list[int] = []
    SCRIPT["select"] = lambda data: []
    SCRIPT["order"] = lambda choices, data: asked.append(1) or []
    assert _refused(
        offline, tmp_path, selector="scripted", orderer="scripted-order"
    ) == (
        "selector scripted chose no field in any well (1 single cells found); "
        "check its parameters, or choose another selector"
    )
    assert asked == []
    # With no wanted cell at all, the first text, whatever the selector.
    assert _refused(offline, tmp_path, selector="scripted", labels=["doublet"]) == (
        "the cells stage found no doublet cell: there is nothing to route; "
        "choose other labels, or check the scan and the cells"
    )


# --------------------------------------------------------------- nis-export


def test_nis_export_exports_a_route_from_a_selector_unchanged(
    tmp_path: Path, offline: Microscope
) -> None:
    a2 = CENTRES["A2"]
    layout = [("A1", A1, 0.0), ("A1", A1, 600.0), ("A2", a2, 0.0)]
    cells = [cell_at(well, dx, 0.0, "single") for well, _, dx in layout]
    _seed(tmp_path, cells, wells=("A1", "A2"))
    SCRIPT["select"] = lambda data: [
        Choice(w, c[0] + dx, c[1], score=0.5, measures={"density_per_mm2": 3.0})
        for w, c, dx in layout
    ]
    SCRIPT["order"] = lambda choices, data: [2, 0, 1]
    route = _result(
        _run(offline, tmp_path, selector="scripted", orderer="scripted-order")
    )

    record = run_plugin(
        NisExportPlugin(), offline, NisExportParams(), tmp_path / "run.json"
    )

    assert record.status is StageStatus.done
    assert record.result is not None
    exported = NisExportResult.model_validate(record.result)
    points = decode((tmp_path / exported.file).read_bytes())
    # The positions in `order`, by name and place; the file holds no score.
    assert [(p.name, p.x_um, p.y_um) for p in points] == [
        (s.name, s.x_um, s.y_um) for s in route.positions
    ]
    assert [p.name for p in points] == ["A2_r000", "A1_r000", "A1_r001"]
    # NIS-Elements' list is UTF-16.
    text = (tmp_path / exported.file).read_bytes().decode("utf-16")
    assert "score" not in text
    assert "density" not in text
