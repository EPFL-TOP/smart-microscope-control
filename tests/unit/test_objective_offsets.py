"""The plate in another objective's frame, on a stub turret and no stand (m3 §18.2, #340).

The shift between two objectives is the profile's, handed out by
``ObjectiveTurret.offset()`` (m2 §13). Here the turret is a stub that holds the
offsets of the design's worked example: the 20X is the reference, the 10X is at
``(5.0, -2.5)`` and the 40X at ``(-32.5, 9.75)``, so the shift 10X -> 40X is
``(-37.5, 12.25)``, anisotropic and far above any tolerance (FM-73). Every
expected number is written here, from those, and never read from the code
under test (FM-124).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smc.hardware.capabilities import ObjectiveOffset
from smc.plugins.errors import PluginRunError
from smc.plugins.objectives import (
    PlateInFrame,
    objective_for,
    objective_shift_um,
    plate_for_objective,
)
from smc.plugins.slots import PlateResult

TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
MEASURED = "2026-11-12, session 3 (#400)"

SHIFT_TEN_TO_FORTY = (-37.5, 12.25)

_OFFSETS = {
    TEN: ObjectiveOffset(TEN, TWENTY, (5.0, -2.5), -1.0, MEASURED),
    FORTY: ObjectiveOffset(FORTY, TWENTY, (-32.5, 9.75), 2.0, MEASURED),
    TWENTY: ObjectiveOffset(TWENTY, TWENTY, (0.0, 0.0), 0.0, ""),
}


def _unknown(label: str) -> ObjectiveOffset:
    return ObjectiveOffset(label, TWENTY, None, None, "")


class _Turret:
    """A turret that holds offsets and answers ``labels`` and ``current``; it never turns."""

    def __init__(
        self,
        offsets: dict[str, ObjectiveOffset] | None = None,
        *,
        labels: tuple[str, ...] = (FORTY, TEN, "", TWENTY),
        current: str = TEN,
    ) -> None:
        self._offsets = _OFFSETS if offsets is None else offsets
        self._labels = list(labels)
        self._current = current
        self.offset_calls: list[str] = []

    def labels(self) -> list[str]:
        return list(self._labels)

    def current(self) -> str:
        return self._current

    def select(self, label: str, *, confirm: bool = False) -> str:
        raise AssertionError("the plate's frame never turns the turret")

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0

    def offset(self, label: str) -> ObjectiveOffset:
        self.offset_calls.append(label)
        return self._offsets.get(label, _unknown(label))


def _plate(objective: str | None = TEN) -> PlateResult:
    """A 96-well plate registered by two references and a lattice's notes, at a stated objective."""
    return PlateResult.model_validate(
        {
            "assumed": ["well pitch from the plate definition"],
            "plate": "96-well",
            "a1_center_xy_um": [1234.5, -987.6],
            "rotation_deg": 0.25,
            "method": "references",
            "residual_um": 12.5,
            "scale_error": 0.002,
            "n_refs": 2,
            "refs": [
                {"well": "A1", "x_um": 1234.5, "y_um": -987.6},
                {"well": "H12", "x_um": 100000.0, "y_um": -63000.0},
            ],
            "fit": {
                "score": 0.93,
                "polarity": 1,
                "runner_up": "B2",
                "margin": 0.12,
                "n_samples": 40,
                "n_inside": 31,
                "notes": ["two samples on the rim"],
            },
            "objective": objective,
        }
    )


# -- objective_shift_um -----------------------------------------------------------


def test_the_shift_is_target_minus_origin() -> None:
    # Neither offset is zero, so swapping the two arguments changes both axes.
    assert objective_shift_um(_OFFSETS[TEN], _OFFSETS[FORTY]) == SHIFT_TEN_TO_FORTY
    assert objective_shift_um(_OFFSETS[FORTY], _OFFSETS[TEN]) == (37.5, -12.25)


def test_the_shift_from_or_to_the_reference_is_the_other_offset() -> None:
    assert objective_shift_um(_OFFSETS[TWENTY], _OFFSETS[FORTY]) == (-32.5, 9.75)
    assert objective_shift_um(_OFFSETS[FORTY], _OFFSETS[TWENTY]) == (32.5, -9.75)


def test_the_shift_is_unknown_when_either_offset_is() -> None:
    assert objective_shift_um(_unknown(TEN), _OFFSETS[FORTY]) is None
    assert objective_shift_um(_OFFSETS[TEN], _unknown(FORTY)) is None
    assert objective_shift_um(_unknown(TEN), _unknown(FORTY)) is None


def test_the_shift_is_unknown_across_references() -> None:
    # Two offsets from different references say nothing about each other.
    other = ObjectiveOffset(FORTY, TEN, (-37.5, 12.25), 0.0, MEASURED)
    assert objective_shift_um(_OFFSETS[TEN], other) is None
    no_reference = ObjectiveOffset(FORTY, None, (-37.5, 12.25), 0.0, MEASURED)
    assert objective_shift_um(_OFFSETS[TEN], no_reference) is None


# -- plate_for_objective: the five rules, in order ----------------------------------


def test_the_same_objective_is_the_plate_itself_with_no_line() -> None:
    plate = _plate(TEN)
    turret = _Turret()
    frame = plate_for_objective(plate, TEN, turret)
    assert isinstance(frame, PlateInFrame)
    assert frame.plate is plate
    assert (frame.objective, frame.shift_um, frame.assumed) == (TEN, (0.0, 0.0), ())
    # Nothing was asked of the profile: the same objective is the same frame.
    assert turret.offset_calls == []


def test_a_plate_and_a_stand_that_both_do_not_know_are_the_same_frame() -> None:
    plate = _plate(None)
    frame = plate_for_objective(plate, None, None)
    assert frame.plate is plate
    assert (frame.objective, frame.shift_um, frame.assumed) == (None, (0.0, 0.0), ())


def test_a_plate_without_objective_is_not_shifted_and_says_so() -> None:
    plate = _plate(None)
    frame = plate_for_objective(plate, FORTY, _Turret())
    assert frame.plate is plate
    assert (frame.objective, frame.shift_um) == (FORTY, None)
    assert frame.assumed == (
        "the plate does not record the objective it was registered with: no offset "
        f"between objectives is applied for {FORTY}",
    )


def test_an_unknown_objective_in_use_is_not_shifted_and_says_so() -> None:
    plate = _plate(TEN)
    frame = plate_for_objective(plate, None, _Turret())
    assert frame.plate is plate
    assert (frame.objective, frame.shift_um) == (None, None)
    assert frame.assumed == (
        f"the objective in use is unknown: the plate's positions are {TEN}'s, and "
        "no offset between objectives is applied",
    )


def test_an_unmeasured_offset_is_not_shifted_and_says_so() -> None:
    plate = _plate(TEN)
    line = (
        f"the offset between {TEN} and {FORTY} is not measured (the profile's "
        f"[objectives]): the positions are {TEN}'s, not corrected"
    )
    for offsets in (
        {TEN: _OFFSETS[TEN]},  # the target's is not measured
        {FORTY: _OFFSETS[FORTY]},  # the origin's is not
        {},  # neither
    ):
        frame = plate_for_objective(plate, FORTY, _Turret(offsets))
        assert frame.plate is plate
        assert (frame.objective, frame.shift_um, frame.assumed) == (
            FORTY,
            None,
            (line,),
        )


def test_a_stand_without_a_turret_cannot_measure_the_offset_and_says_so() -> None:
    plate = _plate(TEN)
    frame = plate_for_objective(plate, FORTY, None)
    assert frame.plate is plate
    assert (frame.objective, frame.shift_um) == (FORTY, None)
    assert frame.assumed == (
        f"the offset between {TEN} and {FORTY} is not measured (the profile's "
        f"[objectives]): the positions are {TEN}'s, not corrected",
    )


def test_the_translated_plate_moves_a1_and_every_ref_and_keeps_the_rotation() -> None:
    plate = _plate(TEN)
    frame = plate_for_objective(plate, FORTY, _Turret())
    moved = frame.plate
    assert frame.shift_um == SHIFT_TEN_TO_FORTY
    assert (frame.objective, frame.assumed) == (FORTY, ())
    assert moved is not plate
    assert isinstance(moved, PlateResult)
    # A1 and each reference are the registration's plus (-37.5, +12.25).
    assert moved.a1_center_xy_um == (1234.5 - 37.5, -987.6 + 12.25)
    assert [(r.well, r.x_um, r.y_um) for r in moved.refs] == [
        ("A1", 1234.5 - 37.5, -987.6 + 12.25),
        ("H12", 100000.0 - 37.5, -63000.0 + 12.25),
    ]
    # Everything else is as it was: the shift is a translation (m2 §13.2).
    assert moved.rotation_deg == 0.25
    assert (moved.plate, moved.method, moved.residual_um, moved.scale_error) == (
        "96-well",
        "references",
        12.5,
        0.002,
    )
    assert moved.n_refs == 2
    assert moved.fit == plate.fit
    assert moved.assumed == plate.assumed
    # The frame is the 40X's now, and the plate it came from is untouched.
    assert moved.objective == FORTY
    assert plate.objective == TEN
    assert plate.a1_center_xy_um == (1234.5, -987.6)


def test_every_well_of_the_translated_plate_is_shifted_by_the_same_amount() -> None:
    plate = _plate(TEN)
    moved = plate_for_objective(plate, FORTY, _Turret()).plate
    for well in ("A1", "D7", "H12"):
        x_um, y_um = plate.well_center_um(well)
        assert moved.well_center_um(well) == pytest.approx(
            (x_um - 37.5, y_um + 12.25), abs=1e-6
        )


def test_the_translated_plate_is_validated_and_not_copied() -> None:
    # A stub that is not the profile hands out an offset the profile would have
    # refused: the plate built from it must be refused too, not carry the inf.
    turret = _Turret(
        {
            TEN: _OFFSETS[TEN],
            FORTY: ObjectiveOffset(FORTY, TWENTY, (float("inf"), 0.0), 0.0, MEASURED),
        }
    )
    with pytest.raises(ValidationError):
        plate_for_objective(_plate(TEN), FORTY, turret)


def test_the_shift_is_asked_of_the_plates_objective_then_the_one_in_use() -> None:
    turret = _Turret()
    plate_for_objective(_plate(TEN), FORTY, turret)
    assert turret.offset_calls == [TEN, FORTY]


# -- objective_for: moved from scan-plan, unchanged -----------------------------------


def test_objective_for_takes_the_one_in_place_when_none_is_asked() -> None:
    assert objective_for("scan-plan", None, _Turret(current=TEN)) == (TEN, True)
    # No turret: the objective is unknown, and the camera's own size is the only one.
    assert objective_for("scan-plan", None, None) == (None, True)


def test_objective_for_says_whether_the_asked_objective_is_the_one_in_place() -> None:
    turret = _Turret(current=TEN)
    assert objective_for("scan-plan", FORTY, turret) == (FORTY, False)
    assert objective_for("scan-plan", TEN, turret) == (TEN, True)


def test_objective_for_refuses_an_empty_position() -> None:
    with pytest.raises(PluginRunError) as refused:
        objective_for("goto-well", None, _Turret(current=""))
    assert refused.value.plugin == "goto-well"
    assert str(refused.value).endswith(
        "no objective is in place: the turret is at an empty position; turn it to "
        "an objective, or give objective to plan for one"
    )


def test_objective_for_refuses_an_objective_that_is_not_on_the_turret() -> None:
    with pytest.raises(PluginRunError) as refused:
        objective_for("plate-registration", "Nikon 100X", _Turret())
    assert str(refused.value).endswith(
        f"objective 'Nikon 100X' is not on the turret: {FORTY}, {TEN}, {TWENTY}"
    )


def test_objective_for_refuses_an_objective_without_a_turret() -> None:
    with pytest.raises(PluginRunError) as refused:
        objective_for("plate-registration", TEN, None)
    assert str(refused.value).endswith(
        "the stand has no turret, so objective cannot be checked; leave it unset"
    )
