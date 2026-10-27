"""The slot models (m3 §3): what every stage reads and writes must round-trip and refuse the unknown.

One table of samples drives the checks that must hold for every model, so that
a model added later cannot dodge them: the first test fails until it has a row.
A sample is the JSON form of a result, written by hand: ``full`` has every
optional field filled and ``sparse`` has every optional ``None``.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import useq
from pydantic import BaseModel, ValidationError

import smc
from smc.plugins import slots
from smc.plugins.slots import (
    CameraCalibration,
    Cell,
    CellsFile,
    CellsResult,
    DensityResult,
    DensityWell,
    FailedField,
    LatticeFit,
    NisExportResult,
    PlannedField,
    PlateResult,
    ReportResult,
    RoutePosition,
    RouteResult,
    ScanFrame,
    ScanPlanResult,
    ScanResult,
    SlotValue,
    TimingsResult,
    WallTouch,
    WellRef,
)

Sample = tuple[dict[str, Any], dict[str, Any]]  # (full, sparse)

_MATRIX = [[0.275, 0.0], [0.0, -0.275]]

_REF = {"well": "A1", "x_um": 1234.5, "y_um": -987.6}
_FIT = {
    "score": 0.93,
    "polarity": 1,
    "runner_up": "B2",
    "margin": 0.12,
    "n_samples": 40,
    "n_inside": 31,
    "notes": ["two samples on the rim"],
}
_PLATE: dict[str, Any] = {
    "assumed": ["well pitch from the plate definition"],
    "plate": "96-well",
    "a1_center_xy_um": [1234.5, -987.6],
    "rotation_deg": 0.25,
    "method": "references",
    "residual_um": 12.5,
    "scale_error": 0.002,
    "n_refs": 2,
    "refs": [_REF, {"well": "H12", "x_um": 100000.0, "y_um": -63000.0}],
    "fit": _FIT,
}

_CALIBRATION: dict[str, Any] = {
    "assumed": ["rotation of the stage taken as 0"],
    "objective": "Nikon 40X Plan Fluor ELWD",
    "image_shape_px": [1200, 1412],
    "matrix_um_per_px": _MATRIX,
    "pixel_size_um": 0.275,
    "rotation_deg": 1.5,
    "mirrored": False,
    "residual_um": 0.05,
    "response": 0.8,
    "return_error_um": 0.2,
    "step_um": 20.0,
    "reused_from": "camera-calibration of 2026-10-20",
}
_PLANNED = {
    "well": "A1",
    "index": 3,
    "name": "A1_0003",
    "x_um": 1100.5,
    "y_um": -990.25,
}
_PLATE_PLAN = useq.WellPlatePlan(
    plate=useq.WellPlate.from_str("96-well"),
    a1_center_xy=(1000.0, 2000.0),
    rotation=0.5,
    selected_wells=([1, 0], [2, 2]),
    well_points_plan=useq.GridRowsColumns(
        rows=2, columns=3, overlap=(10.0, 10.0), fov_width=388.3, fov_height=330.0
    ),
).model_dump(mode="json")
_SCAN_PLAN: dict[str, Any] = {
    "assumed": [],
    "plate_plan": _PLATE_PLAN,
    "wells": ["B3", "A3"],
    "objective": "Nikon 40X Plan Fluor ELWD",
    "pixel_size_um": 0.275,
    "matrix_um_per_px": _MATRIX,
    "matrix_source": "camera-calibration",
    "image_shape_px": [1200, 1412],
    "fov_um": [388.3, 330.0],
    "overlap": 0.1,
    "rows": 2,
    "columns": 3,
    "fields": [_PLANNED],
    "estimate_s": 1830.5,
}

_FRAME: dict[str, Any] = {
    "well": "A1",
    "index": 0,
    "name": "A1_0000",
    "file": "scan/0123456789ab/A1_0000.tif",
    "x_um": 1000.5,
    "y_um": 2000.25,
    "z_um": 1500.0,
    "focus": "locked",
    "pfs_offset": 512.0,
    "t_s": 12.5,
}
_FAILED = {
    "well": "A1",
    "index": 1,
    "name": "A1_0001",
    "x_um": 1100.0,
    "y_um": 2000.0,
    "error": "HardwareError: the stage did not answer",
}
_SCAN: dict[str, Any] = {
    "assumed": [],
    "folder": "scan/0123456789ab",
    "objective": "Nikon 40X Plan Fluor ELWD",
    "pixel_size_um": 0.275,
    "matrix_um_per_px": _MATRIX,
    "image_shape_px": [1200, 1412],
    "exposure_ms": 10.0,
    "frames": [_FRAME],
    "failed": [_FAILED],
}

_CELL: dict[str, Any] = {
    "well": "A1",
    "x_um": 1010.5,
    "y_um": 2010.25,
    "label": "single",
    "score": 0.91,
    "frame": 0,
    "box_px": [100.0, 200.0, 140.5, 240.5],
}
_CELLS_FILE = {"schema_version": 1, "cells": [_CELL]}
_CELLS: dict[str, Any] = {
    "assumed": [],
    "file": "cells/0123456789ab.json",
    "model": "cell_classes_model-cat-v2.pth",
    "model_sha256": "0" * 64,
    "classes": ["single", "doublet", "debris"],
    "device": "cpu",
    "score_threshold": 0.5,
    "merge_radius_um": 8.0,
    "frames": 12,
    "counts": {"A1": {"single": 1, "doublet": 0, "debris": 0}},
    "merged": 3,
    "seconds_per_frame": 0.34,
}

_DENSITY_WELL: dict[str, Any] = {
    "well": "A1",
    "fields": 305,
    "failed_fields": 2,
    "area_mm2": 31.2,
    "counts": {"single": 40, "doublet": 3},
    "outside_well": 5,
    "per_mm2": {"single": 1.28, "doublet": 0.09},
}
_DENSITY: dict[str, Any] = {
    "assumed": [],
    "labels": ["single", "doublet"],
    "area_grid_um": 50.0,
    "wells": [_DENSITY_WELL],
}
_STOP: dict[str, Any] = {
    "order": 0,
    "well": "A1",
    "name": "A1_0003",
    "x_um": 1100.5,
    "y_um": -990.25,
    "z_um": 1500.0,
    "pfs_offset": 512.0,
    "counts": {"single": 2},
}
_ROUTE: dict[str, Any] = {
    "assumed": [],
    "objective": "Nikon 40X Plan Fluor ELWD",
    "matrix_um_per_px": _MATRIX,
    "image_shape_px": [1200, 1412],
    "fov_um": [388.3, 330.0],
    "labels": ["single"],
    "positions": [_STOP],
    "uncovered": {"A1": 0},
    "distance_um": 9000.0,
    "travel_s": 1.5,
}
_TIMINGS: dict[str, Any] = {
    "assumed": ["refocus_ms assumed from the autofocus datasheet"],
    "move_overhead_ms": 30.0,
    "move_ms_per_mm": 12.5,
    "refocus_ms": 250.0,
    "snap_ms": 100.0,
    "channel_ms": {"DAPI": 120.0},
    "channel_switch_ms": 80.0,
    "samples": 8,
}
_REPORT = {"assumed": [], "file": "report/0123456789ab.html"}
_NIS = {
    "assumed": [],
    "file": "nis/0123456789ab.xml",
    "positions": 12,
    "z_mode": "pfs-offset",
}

#: Every model of ``smc.plugins.slots`` with a full and a sparse sample. A module
#: that lands later adds its rows here (the first test fails until it does).
SAMPLES: dict[type[BaseModel], Sample] = {
    WellRef: (_REF, _REF),
    WallTouch: (
        {"well": "A1", "side": "left", "x_um": 1200.5, "y_um": -980.0},
        {"well": "A1", "side": "left", "x_um": 1200.5, "y_um": -980.0},
    ),
    LatticeFit: (
        _FIT,
        {
            "score": 0.5,
            "polarity": -1,
            "runner_up": None,
            "margin": None,
            "n_samples": 0,
            "n_inside": 0,
        },
    ),
    PlateResult: (
        _PLATE,
        {
            "plate": "384-well",
            "a1_center_xy_um": [0.0, 0.0],
            "rotation_deg": 0.0,
            "method": "seed",
            "residual_um": None,
            "scale_error": None,
            "n_refs": 0,
        },
    ),
    CameraCalibration: (
        _CALIBRATION,
        {
            **_CALIBRATION,
            "objective": None,
            "return_error_um": None,
            "reused_from": None,
        },
    ),
    PlannedField: (_PLANNED, _PLANNED),
    ScanPlanResult: (
        _SCAN_PLAN,
        {
            **_SCAN_PLAN,
            "objective": None,
            "matrix_source": "profile",
            "wells": [],
            "fields": [],
            "estimate_s": None,
        },
    ),
    ScanFrame: (
        _FRAME,
        {**_FRAME, "z_um": None, "focus": "none", "pfs_offset": None},
    ),
    FailedField: (_FAILED, _FAILED),
    ScanResult: (
        _SCAN,
        {**_SCAN, "objective": None, "frames": [], "failed": []},
    ),
    Cell: (_CELL, _CELL),
    CellsFile: (_CELLS_FILE, {"cells": []}),
    CellsResult: (
        _CELLS,
        {**_CELLS, "counts": {}, "seconds_per_frame": None},
    ),
    DensityWell: (
        _DENSITY_WELL,
        {**_DENSITY_WELL, "area_mm2": 0.0, "counts": {}, "per_mm2": None},
    ),
    DensityResult: (_DENSITY, {**_DENSITY, "wells": []}),
    RoutePosition: (_STOP, {**_STOP, "z_um": None, "pfs_offset": None}),
    RouteResult: (
        _ROUTE,
        {**_ROUTE, "objective": None, "positions": [], "travel_s": None},
    ),
    TimingsResult: (
        _TIMINGS,
        {
            **{
                k: v
                for k, v in _TIMINGS.items()
                if k not in ("channel_ms", "channel_switch_ms")
            },
            "refocus_ms": None,
        },
    ),
    ReportResult: (_REPORT, _REPORT),
    NisExportResult: (_NIS, {**_NIS, "positions": 0, "z_mode": "none"}),
}

#: Models that hold no float at all, so the finite-number walk has nothing to hit.
NO_FLOAT: set[type[BaseModel]] = {ReportResult, NisExportResult}


def _leaves(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Every scalar of a JSON-like value with its path (dict keys and list indices)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _with(sample: dict[str, Any], path: tuple[Any, ...], leaf: Any) -> dict[str, Any]:
    """A deep copy of ``sample`` with the value at ``path`` replaced; the sample is not touched."""
    changed = copy.deepcopy(sample)
    node: Any = changed
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = leaf
    return changed


def _without(sample: dict[str, Any], path: tuple[Any, ...]) -> dict[str, Any]:
    changed = copy.deepcopy(sample)
    node: Any = changed
    for step in path[:-1]:
        node = node[step]
    del node[path[-1]]
    return changed


def _error_types(model: type[BaseModel], data: dict[str, Any]) -> list[str]:
    with pytest.raises(ValidationError) as raised:
        model.model_validate(data)
    return [error["type"] for error in raised.value.errors()]


def _exported_models() -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = set()
    for name in slots.__all__:
        item = getattr(slots, name)
        if isinstance(item, type) and issubclass(item, BaseModel):
            found.add(item)
    # SlotValue is the base of the values, with no field of its own to sample.
    return found - {SlotValue}


MODELS = pytest.mark.parametrize("model", list(SAMPLES), ids=lambda m: m.__name__)


# --- what every model must do ----------------------------------------------------


def test_every_exported_slot_model_has_a_sample() -> None:
    assert _exported_models() == set(SAMPLES)


@MODELS
def test_every_slot_model_round_trips_through_json(model: type[BaseModel]) -> None:
    full, sparse = SAMPLES[model]
    for sample in (full, sparse):
        original = model.model_validate(sample)
        assert model.model_validate(original.model_dump(mode="json")) == original
    # The full sample must really be full, or the round trip proves little.
    full_dump = model.model_validate(full).model_dump(mode="json")
    assert [path for path, leaf in _leaves(full_dump) if leaf is None] == []


@MODELS
def test_every_float_of_every_slot_model_refuses_nan_and_inf(
    model: type[BaseModel],
) -> None:
    full, _ = SAMPLES[model]
    dump = model.model_validate(full).model_dump(mode="json")
    floats = [path for path, leaf in _leaves(dump) if isinstance(leaf, float)]
    if model in NO_FLOAT:
        assert floats == []
        return
    assert floats, (
        f"{model.__name__} has no float leaf: add it to NO_FLOAT or fix the sample"
    )
    for path in floats:
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValidationError):
                model.model_validate(_with(dump, path, bad))


@MODELS
def test_every_slot_model_refuses_an_unknown_key(model: type[BaseModel]) -> None:
    # FM-107: a key this smc does not know is refused, never dropped and rewritten.
    full, _ = SAMPLES[model]
    types = _error_types(model, {**full, "bogus_key": 1})
    assert types == ["extra_forbidden"]


@MODELS
def test_every_slot_model_is_frozen(model: type[BaseModel]) -> None:
    # A result read from the run document is not edited in place: the record
    # that holds it says what the run produced.
    full, _ = SAMPLES[model]
    loaded = model.model_validate(full)
    field = next(iter(type(loaded).model_fields))
    with pytest.raises(ValidationError) as raised:
        setattr(loaded, field, getattr(loaded, field))
    assert [error["type"] for error in raised.value.errors()] == ["frozen_instance"]


# --- ranges ----------------------------------------------------------------------

#: (model, path in the full sample, values refused, values accepted). Each refused
#: value differs from the bound it is compared with, and each accepted one is on
#: it (or, for ``gt``, as close above it as a test can be).
RANGES: list[tuple[type[BaseModel], tuple[Any, ...], list[Any], list[Any]]] = [
    (LatticeFit, ("score",), [-0.1, 1.5], [0.0, 1.0]),
    (LatticeFit, ("n_samples",), [-1], [0]),
    (LatticeFit, ("n_inside",), [-1], [0]),
    (PlateResult, ("rotation_deg",), [-180.5, 180.5], [-180.0, 180.0]),
    (PlateResult, ("residual_um",), [-0.1], [0.0]),
    (PlateResult, ("n_refs",), [-1], [0]),
    (CameraCalibration, ("pixel_size_um",), [0.0, -1.0], [1e-6]),
    (CameraCalibration, ("rotation_deg",), [-180.5, 180.5], [-180.0, 180.0]),
    (CameraCalibration, ("residual_um",), [-0.1], [0.0]),
    (CameraCalibration, ("response",), [-0.1, 1.5], [0.0, 1.0]),
    (CameraCalibration, ("return_error_um",), [-0.1], [0.0]),
    (CameraCalibration, ("step_um",), [0.0, -1.0], [1e-6]),
    (PlannedField, ("index",), [-1], [0]),
    (ScanPlanResult, ("pixel_size_um",), [0.0, -1.0], [1e-6]),
    (ScanPlanResult, ("overlap",), [-0.1, 1.0, 1.5], [0.0, 0.999]),
    (ScanPlanResult, ("rows",), [0, -1], [1]),
    (ScanPlanResult, ("columns",), [0, -1], [1]),
    (ScanPlanResult, ("estimate_s",), [-0.1], [0.0]),
    (ScanFrame, ("index",), [-1], [0]),
    (ScanFrame, ("t_s",), [-0.1], [0.0]),
    (FailedField, ("index",), [-1], [0]),
    (ScanResult, ("pixel_size_um",), [0.0, -1.0], [1e-6]),
    (ScanResult, ("exposure_ms",), [-0.1], [0.0]),
    (Cell, ("score",), [-0.1, 1.5], [0.0, 1.0]),
    (Cell, ("frame",), [-1], [0]),
    (CellsResult, ("frames",), [-1], [0]),
    (CellsResult, ("merged",), [-1], [0]),
    (CellsResult, ("seconds_per_frame",), [-0.1], [0.0]),
    (DensityWell, ("fields",), [-1], [0]),
    (DensityWell, ("failed_fields",), [-1], [0]),
    (DensityWell, ("area_mm2",), [-0.1], [0.0]),
    (DensityWell, ("outside_well",), [-1], [0]),
    (DensityResult, ("area_grid_um",), [0.0, -1.0], [1e-6]),
    (RoutePosition, ("order",), [-1], [0]),
    (RouteResult, ("distance_um",), [-0.1], [0.0]),
    (RouteResult, ("travel_s",), [-0.1], [0.0]),
    (TimingsResult, ("move_overhead_ms",), [-0.1], [0.0]),
    (TimingsResult, ("move_ms_per_mm",), [-0.1], [0.0]),
    (TimingsResult, ("refocus_ms",), [-0.1], [0.0]),
    (TimingsResult, ("snap_ms",), [-0.1], [0.0]),
    (TimingsResult, ("channel_switch_ms",), [-0.1], [0.0]),
    (TimingsResult, ("samples",), [-1], [0]),
    (NisExportResult, ("positions",), [-1], [0]),
]


def _range_cases(index: int) -> list[Any]:
    cases = []
    for model, path, refused, accepted in RANGES:
        values = (refused, accepted)[index]
        cases.extend(
            pytest.param(
                model,
                path,
                value,
                id=f"{model.__name__}.{'.'.join(map(str, path))}={value}",
            )
            for value in values
        )
    return cases


@pytest.mark.parametrize(("model", "path", "value"), _range_cases(0))
def test_a_value_outside_its_range_is_refused(
    model: type[BaseModel], path: tuple[Any, ...], value: Any
) -> None:
    full, _ = SAMPLES[model]
    with pytest.raises(ValidationError) as raised:
        model.model_validate(_with(full, path, value))
    # One error, about this field, and about its range (not a type or a nan).
    (error,) = raised.value.errors()
    assert error["loc"] == path
    assert error["type"] in {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
    }


@pytest.mark.parametrize(("model", "path", "value"), _range_cases(1))
def test_a_value_on_its_bound_is_accepted(
    model: type[BaseModel], path: tuple[Any, ...], value: Any
) -> None:
    full, _ = SAMPLES[model]
    model.model_validate(_with(full, path, value))


# --- a field that may be unknown must still be present ---------------------------

#: Required fields whose value may be ``None``: a producer must say it does not know.
MUST_BE_PRESENT: list[tuple[type[BaseModel], tuple[Any, ...]]] = [
    (PlateResult, ("residual_um",)),
    (PlateResult, ("scale_error",)),
    (LatticeFit, ("runner_up",)),
    (LatticeFit, ("margin",)),
    (CameraCalibration, ("objective",)),
    (CameraCalibration, ("return_error_um",)),
    (ScanPlanResult, ("objective",)),
    (ScanPlanResult, ("estimate_s",)),
    (ScanFrame, ("z_um",)),
    (ScanFrame, ("pfs_offset",)),
    (ScanResult, ("objective",)),
    (CellsResult, ("seconds_per_frame",)),
    (DensityWell, ("per_mm2",)),
    (RoutePosition, ("z_um",)),
    (RoutePosition, ("pfs_offset",)),
    (RouteResult, ("objective",)),
    (RouteResult, ("travel_s",)),
    (TimingsResult, ("refocus_ms",)),
]


@pytest.mark.parametrize(
    ("model", "path"),
    MUST_BE_PRESENT,
    ids=lambda p: p.__name__ if isinstance(p, type) else ".".join(map(str, p)),
)
def test_a_field_that_may_be_unknown_must_still_be_present(
    model: type[BaseModel], path: tuple[Any, ...]
) -> None:
    full, _ = SAMPLES[model]
    assert _error_types(model, _without(full, path)) == ["missing"]
    model.model_validate(_with(full, path, None))


# --- the plate -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["glass", "96-WELL", ""])
def test_plate_refuses_a_plate_that_useq_does_not_know(name: str) -> None:
    with pytest.raises(ValidationError) as raised:
        PlateResult.model_validate({**_PLATE, "plate": name})
    errors = raised.value.errors()
    assert [(error["type"], error["loc"]) for error in errors] == [
        ("value_error", ("plate",))
    ]


def test_a_subclass_of_a_slot_model_is_refused_through_the_base() -> None:
    # §2.3: a second provider of a slot returns the slot's model itself, because
    # the dump of a subclass that adds a field is refused when read as its base.
    class Fancy(PlateResult):
        extra_field: int = 1

    dump = Fancy.model_validate(_PLATE).model_dump(mode="json")
    assert dump["extra_field"] == 1
    assert _error_types(PlateResult, dump) == ["extra_forbidden"]


def test_to_plan_carries_the_registration() -> None:
    result = PlateResult.model_validate(_PLATE)
    plan = result.to_plan(selected_wells=([1, 0], [0, 5]))
    assert plan.a1_center_xy == (1234.5, -987.6)
    assert plan.rotation == 0.25
    assert plan.plate == useq.WellPlate.from_str("96-well")
    assert [str(name) for name in plan.selected_well_names] == ["B1", "A6"]


def test_well_center_um_normalises_its_argument_and_refuses_a_well_not_on_the_plate() -> (
    None
):
    result = PlateResult.model_validate({**_PLATE, "rotation_deg": 0.0})
    # No rotation: A2 is one pitch (9 mm) to the right of A1, B1 one pitch below.
    assert result.well_center_um("a02") == pytest.approx((1234.5 + 9000.0, -987.6))
    assert result.well_center_um(" B1 ") == pytest.approx((1234.5, -987.6 - 9000.0))
    assert result.well_center_um("A1") == (1234.5, -987.6)
    with pytest.raises(ValueError, match=r"\Awell 'I13' is not on the 96-well plate\Z"):
        result.well_center_um("I13")


# --- the scan ------------------------------------------------------------------


def test_a_focus_state_the_scan_does_not_know_is_refused() -> None:
    assert _error_types(ScanFrame, {**_FRAME, "focus": "maybe"}) == ["literal_error"]
    for state in ("locked", "unlocked", "off", "none"):
        ScanFrame.model_validate({**_FRAME, "focus": state})


def test_frame_index_maps_a_well_to_the_positions_of_its_frames() -> None:
    def frame(well: str, index: int) -> dict[str, Any]:
        return {**_FRAME, "well": well, "index": index, "name": f"{well}_{index:04d}"}

    # Frames in the order A1, B1, A1: a well's positions are not contiguous, the
    # wells come in the order of their first frame, and a well with no frame
    # (every field of C1 failed) is absent.
    result = ScanResult.model_validate(
        {**_SCAN, "frames": [frame("A1", 0), frame("B1", 0), frame("A1", 1)]}
    )
    index = result.frame_index()
    assert index == {"A1": [0, 2], "B1": [1]}
    assert list(index) == ["A1", "B1"]
    assert "C1" not in index
    assert ScanResult.model_validate({**_SCAN, "frames": []}).frame_index() == {}


# --- timings, the route, and what the foundation may import ----------------------


def test_move_ms_is_zero_for_no_distance_and_linear_beyond_it() -> None:
    timings = TimingsResult.model_validate(_TIMINGS)  # 30 ms overhead, 12.5 ms/mm
    assert timings.move_ms(0.0) == 0.0
    assert timings.move_ms(-5.0) == 0.0
    assert timings.move_ms(1000.0) == pytest.approx(42.5)
    assert timings.move_ms(2000.0) == pytest.approx(55.0)
    # A nan would slip past `<= 0` and give a nan time that every sum then carries.
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(
            ValueError,
            match=rf"\Adistance_um must be finite, got {re.escape(repr(bad))}\Z",
        ):
            timings.move_ms(bad)


def test_route_to_useq_keeps_the_route_order_and_an_unknown_z() -> None:
    stops = [
        {
            **_STOP,
            "order": 0,
            "name": "B1_0000",
            "x_um": 5.0,
            "y_um": 6.0,
            "z_um": None,
        },
        {**_STOP, "order": 1, "name": "A1_0002", "x_um": 1.0, "y_um": 2.0, "z_um": 3.0},
    ]
    route = RouteResult.model_validate({**_ROUTE, "positions": stops})
    positions = route.to_useq()
    assert all(isinstance(p, useq.AbsolutePosition) for p in positions)
    # The order of the route, not of the names, and a z that is unknown stays so.
    assert [(p.name, p.x, p.y, p.z) for p in positions] == [
        ("B1_0000", 5.0, 6.0, None),
        ("A1_0002", 1.0, 2.0, 3.0),
    ]
    assert RouteResult.model_validate({**_ROUTE, "positions": []}).to_useq() == ()


def _imports_of(path: Path) -> set[str]:
    """Every module a source file imports, by its dotted name.

    ``from smc import hardware`` imports ``smc.hardware`` as much as
    ``import smc.hardware`` does, so each name of a ``from`` is recorded under
    its module. A relative import is recorded with its dots, and the test below
    refuses any: the project imports absolutely, and a dotted name that starts
    with a dot would slip past a check for ``smc.hardware``.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = ("." * node.level) + (node.module or "")
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_the_foundation_imports_no_hardware_and_files_imports_no_plugins() -> None:
    # ADR-0003: hardware is reached only through smc.hardware, by the layers that
    # own a device. The slot models and the geometry are arithmetic on numbers.
    root = Path(smc.__file__).parent
    foundation = [root / "plugins" / "geometry.py", root / "run" / "files.py"]
    foundation += sorted((root / "plugins" / "slots").glob("*.py"))
    assert len(foundation) >= 14  # geometry, files, and the twelve slot modules
    for path in foundation:
        for name in _imports_of(path):
            assert not name.startswith("."), (path.name, name)
            assert not name.startswith(("smc.hardware", "pymmcore")), (path.name, name)
    # smc.run is below smc.plugins: plugins depend on the document, not the reverse.
    assert not [n for n in _imports_of(root / "run" / "files.py") if "smc.plugins" in n]
    assert not [
        n for n in _imports_of(root / "run" / "document.py") if "smc.plugins" in n
    ]
    # And smc.plugins never loads the foundation by accident (m3 §3: plugins import
    # what they need from it, a plugin that is merely listed does not).
    package = _imports_of(root / "plugins" / "__init__.py")
    assert not [n for n in package if n.startswith("smc.plugins.slots")]
    assert "smc.plugins.geometry" not in package


#: The names of design §3's map that ``smc.plugins.slots`` re-exports, and that eight
#: parallel issues import: a rename or a drop is a plan deviation, not a quiet edit.
SLOTS_MAP = {
    "CameraCalibration",
    "Cell",
    "CellsFile",
    "CellsResult",
    "DensityResult",
    "DensityWell",
    "FailedField",
    "FocusState",
    "LatticeFit",
    "NisExportResult",
    "PlannedField",
    "PlateResult",
    "ReportResult",
    "RoutePosition",
    "RouteResult",
    "ScanFrame",
    "ScanPlanResult",
    "ScanResult",
    "SlotValue",
    "TimingsResult",
    "WallTouch",
    "WellGeometry",
    "WellRef",
    "load_cells",
    "load_frame",
    "normalise_well",
    "well_geometry",
}


def test_slots_re_exports_every_name_of_the_design_map() -> None:
    assert set(slots.__all__) == SLOTS_MAP
    # What `import *` would bind: every name of `__all__` must resolve.
    assert [name for name in slots.__all__ if not hasattr(slots, name)] == []


def test_the_plugin_package_exports_plugin_run_error() -> None:
    import smc.plugins

    assert "PluginRunError" in smc.plugins.__all__
    from smc.plugins import PluginRunError

    assert PluginRunError is smc.plugins.errors.PluginRunError


@pytest.mark.parametrize(
    "where",
    [(), ("well_points_plan",), ("plate",)],
    ids=["plan", "grid", "plate"],
)
def test_a_key_useq_would_drop_is_refused_inside_the_plate_plan(
    where: tuple[str, ...],
) -> None:
    # FM-107: useq's models ignore keys they do not know, so a key written by a
    # newer smc (or useq) would load, vanish, and be written back without it.
    sample = copy.deepcopy(_SCAN_PLAN)
    node = sample["plate_plan"]
    for step in where:
        node = node[step]
    node["bogus_nested"] = 1
    with pytest.raises(ValidationError) as raised:
        ScanPlanResult.model_validate(sample)
    (error,) = raised.value.errors()
    assert error["type"] == "value_error"
    assert error["loc"] == ("plate_plan",)
    dotted = ".".join(("plate_plan", *where, "bogus_nested"))
    assert error["msg"] == (
        f"Value error, {dotted} is not a key this smc knows; it would be dropped "
        f"(a newer smc wrote it? upgrade smc)"
    )


def test_a_plate_given_by_name_is_not_a_dropped_key() -> None:
    # useq turns "96-well" into a plate: the dump has a dict where the input had a
    # string, and that is a normalisation, not a loss.
    sample = copy.deepcopy(_SCAN_PLAN)
    sample["plate_plan"]["plate"] = "96-well"
    loaded = ScanPlanResult.model_validate(sample)
    assert loaded.plate_plan.plate.name == "96-well"
