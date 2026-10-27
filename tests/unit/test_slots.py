"""The slot models (m3 §3): what every stage reads and writes must round-trip and refuse the unknown.

One table of samples drives the checks that must hold for every model, so that
a model added later cannot dodge them: the first test fails until it has a row.
A sample is the JSON form of a result, written by hand: ``full`` has every
optional field filled and ``sparse`` has every optional ``None``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Any

import pytest
import useq
from pydantic import BaseModel, ValidationError

from smc.plugins import slots
from smc.plugins.slots import (
    CameraCalibration,
    LatticeFit,
    PlannedField,
    PlateResult,
    ScanPlanResult,
    SlotValue,
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
}

#: Models that hold no float at all, so the finite-number walk has nothing to hit.
NO_FLOAT: set[type[BaseModel]] = set()


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
    assert _error_types(model, _with(full, path, value))


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
