"""The cells' plane's offset in a route stop: the pure rule and its texts (m3 §26, #392).

A scan with the focus on the cells takes each frame at the cells' plane with the
autofocus off, and records the offset of the lock it swept from. The rule turns
that distance into the offset NIS-Elements would engage the autofocus at, or says
it cannot. Every expected number is written here from the frame's own numbers and
the factor, never read back from the result (FM-124), and each bound is tested on
both of its sides (FM-121).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import pytest

from smc.plugins.cell_focus.algorithm import MAX_RANGE_UM
from smc.plugins.route.focus import StopFocus, StopOutcome, focus_assumed, stop_focus
from smc.plugins.slots import (
    FieldFocus,
    FocusState,
    FocusSweep,
    OffsetFromFocus,
    ScanFrame,
)

OBJ = "Nikon 40X Plan Fluor ELWD"
FACTORS = {OBJ: 112.6}
LIMITS = (0.0, 20000.0)
Z_UM = 1500.25


def _sweep(outcome: str = "peak") -> FocusSweep:
    return FocusSweep.model_validate(
        {
            "score": "brenner",
            "centre_um": 1500.0,
            "z_um": [1499.0, 1500.0, 1501.0],
            "scores": [0.12, 0.31, 0.2],
            "outcome": outcome,
            "best_z_um": 1500.2 if outcome == "peak" else None,
        }
    )


def _focus(
    dz_um: float, *, source: Literal["swept", "well"] = "well", outcome: str = "peak"
) -> FieldFocus:
    return FieldFocus(
        source=source,
        sweep=_sweep(outcome) if source == "swept" else None,
        dz_um=dz_um,
        frame_score=0.3,
        duration_s=0.25,
    )


def _frame(
    cell_focus: FieldFocus | None,
    *,
    focus: FocusState = "locked",
    pfs_offset: float | None = 5388.0,
    z_um: float | None = Z_UM,
) -> ScanFrame:
    # Not square and away from the origin: an exchange of the axes would show.
    return ScanFrame(
        well="B3",
        index=2,
        name="B3_0002",
        file="scan/0123456789ab/B3_0002.tif",
        x_um=1234.5,
        y_um=-678.25,
        z_um=z_um,
        focus=focus,
        pfs_offset=pfs_offset,
        t_s=1.5,
        cell_focus=cell_focus,
    )


def _of(
    dz_um: float,
    *,
    lock: float = 5388.0,
    objective: str | None = OBJ,
    factors: dict[str, float] | None = None,
    limits: tuple[float, float] | None = None,
) -> StopFocus:
    return stop_focus(
        _frame(_focus(dz_um), pfs_offset=lock),
        objective=objective,
        factors=FACTORS if factors is None else factors,
        offset_limits=limits,
    )


# --- a stop that is left as the scan recorded it ----------------------------------


@pytest.mark.parametrize("factors", [{}, FACTORS], ids=["no-factor", "a-factor"])
def test_a_frame_without_a_focus_step_keeps_the_scans_offset(
    factors: dict[str, float],
) -> None:
    got = stop_focus(_frame(None), objective=OBJ, factors=factors, offset_limits=LIMITS)
    assert got == StopFocus(
        z_um=Z_UM,
        pfs_offset=5388.0,
        derived=None,
        outcome="recorded",
        refused_offset=None,
        no_peak=False,
    )


def test_a_frame_at_the_centre_keeps_the_scans_offset_and_needs_no_factor() -> None:
    # dz_um 0.0: the frame's Z is the lock's plane and its offset names it, so
    # there is no distance to convert and no factor to ask for (D3).
    for objective in (None, OBJ):
        got = stop_focus(
            _frame(_focus(0.0)),
            objective=objective,
            factors={},
            offset_limits=LIMITS,
        )
        assert (got.outcome, got.z_um, got.pfs_offset, got.derived) == (
            "recorded",
            Z_UM,
            5388.0,
            None,
        )


@pytest.mark.parametrize(
    ("focus", "pfs_offset"),
    [("unlocked", 4000.0), ("off", None), ("none", None), ("locked", None)],
)
def test_an_unlocked_off_or_none_frame_is_left_as_recorded(
    focus: FocusState, pfs_offset: float | None
) -> None:
    # FM-154: an unlocked frame's offset is one that nobody read locked on this
    # sample, an `off` or `none` frame has none, and a locked frame without an
    # offset has nothing to add a distance to. None of them is derived, and a
    # distance with a factor beside it does not change that.
    got = stop_focus(
        _frame(_focus(-3.0), focus=focus, pfs_offset=pfs_offset),
        objective=OBJ,
        factors=FACTORS,
        offset_limits=LIMITS,
    )
    assert (got.outcome, got.z_um, got.pfs_offset, got.derived) == (
        "recorded",
        Z_UM,
        pfs_offset,
        None,
    )


def test_a_frame_without_a_frame_has_nothing() -> None:
    # A route in another objective has no frame to read (§11): today's `None`s.
    got = stop_focus(None, objective=OBJ, factors=FACTORS, offset_limits=LIMITS)
    assert got == StopFocus(
        z_um=None,
        pfs_offset=None,
        derived=None,
        outcome="recorded",
        refused_offset=None,
        no_peak=False,
    )


# --- the arithmetic, sign and unit --------------------------------------------------


@pytest.mark.parametrize(
    ("lock", "dz_um", "factor", "expected"),
    [
        # 5388 - 3.0 * 112.6 = 5050.2
        (5388.0, -3.0, 112.6, 5050.0),
        # 5388 + 2.7 * 112.6 = 5692.02
        (5388.0, 2.7, 112.6, 5692.0),
        # A factor with the other sign mirrors the first: 5388 + 3.0 * 112.6 = 5725.8
        (5388.0, -3.0, -112.6, 5726.0),
    ],
)
def test_the_offset_moves_by_the_distance_times_the_factor(
    lock: float, dz_um: float, factor: float, expected: float
) -> None:
    got = _of(dz_um, lock=lock, factors={OBJ: factor})
    assert got.outcome == "derived"
    assert got.pfs_offset == expected
    # The record says what it was derived from: three numbers, and the sign.
    assert got.derived == OffsetFromFocus(
        lock_offset=lock, dz_um=dz_um, offset_per_um=factor
    )
    assert got.z_um == Z_UM
    assert got.refused_offset is None
    assert got.no_peak is False


def test_the_offset_is_rounded_to_a_whole_unit() -> None:
    # 5388 - 2.95 * 112.6 = 5055.83: up, not truncated to 5055. And a float with
    # an integer value, as the export writes it.
    got = _of(-2.95)
    assert got.pfs_offset == 5056.0
    assert isinstance(got.pfs_offset, float)
    assert got.pfs_offset.is_integer()


# --- no offset, and why --------------------------------------------------------------


@pytest.mark.parametrize(
    ("objective", "factors", "outcome"),
    [
        (None, FACTORS, "no-objective"),
        ("Nikon 10X S Fluor", FACTORS, "no-factor"),
        (OBJ, {}, "no-factor"),
        (OBJ, {OBJ: 0.0}, "no-factor"),
        (OBJ, {OBJ: -0.0}, "no-factor"),
        (OBJ, {OBJ: math.nan}, "no-factor"),
        (OBJ, {OBJ: math.inf}, "no-factor"),
        (OBJ, {OBJ: -math.inf}, "no-factor"),
        # A factor so large that the sum overflows is not a factor either.
        (OBJ, {OBJ: 1e308}, "no-factor"),
        # The label is looked up exactly: another objective's factor is never used.
        (OBJ.lower(), FACTORS, "no-factor"),
        (OBJ + " ", FACTORS, "no-factor"),
    ],
)
def test_no_objective_no_entry_or_a_bad_factor_gives_no_offset_and_keeps_z(
    objective: str | None, factors: dict[str, float], outcome: StopOutcome
) -> None:
    # FM-214: a factor measured at one objective is never used for another.
    got = _of(-3.0, objective=objective, factors=factors)
    assert got.outcome == outcome
    assert got.pfs_offset is None
    assert got.derived is None
    assert got.refused_offset is None
    assert got.z_um == Z_UM


@pytest.mark.parametrize(
    ("lock", "dz_um", "factor", "kept", "refused"),
    [
        # 19000 + 10 * 100 = 20000: on the upper limit, kept; one unit more, refused.
        (19000.0, 10.0, 100.0, 20000.0, None),
        (19001.0, 10.0, 100.0, None, 20001.0),
        # 1000 - 10 * 100 = 0: on the lower limit, kept; one unit below, refused.
        (1000.0, -10.0, 100.0, 0.0, None),
        (999.0, -10.0, 100.0, None, -1.0),
    ],
)
def test_an_offset_outside_the_profiles_limits_is_left_out_from_both_sides(
    lock: float,
    dz_um: float,
    factor: float,
    kept: float | None,
    refused: float | None,
) -> None:
    # FM-121: the offset ends in none, never a clamp: NIS-Elements engages the
    # autofocus where it is told, with none of smc's guards (m2 §12.10).
    got = _of(dz_um, lock=lock, factors={OBJ: factor}, limits=LIMITS)
    assert got.pfs_offset == kept
    assert got.refused_offset == refused
    assert got.z_um == Z_UM
    if kept is not None:
        assert got.outcome == "derived"
        assert got.derived is not None
    else:
        assert got.outcome == "outside-limits"
        assert got.derived is None


def test_without_limits_an_offset_beyond_them_is_kept() -> None:
    got = _of(10.0, lock=19001.0, factors={OBJ: 100.0}, limits=None)
    assert (got.outcome, got.pfs_offset) == ("derived", 20001.0)


@pytest.mark.parametrize(
    ("dz_um", "outcome", "offset"),
    [
        # 5388 + 49.9 x 10 = 5887; 5388 - 49.9 x 10 = 4889 (the bound is 50 µm).
        (49.9, "derived", 5887.0),
        (-49.9, "derived", 4889.0),
        (MAX_RANGE_UM, "derived", 5888.0),
        (-MAX_RANGE_UM, "derived", 4888.0),
        (50.1, "implausible", None),
        (-50.1, "implausible", None),
    ],
)
def test_a_distance_beyond_the_sweeps_largest_range_is_implausible(
    dz_um: float, outcome: StopOutcome, offset: float | None
) -> None:
    # The scan never sweeps wider than 50 µm: a larger distance is a record
    # edited behind the runner (FM-111), not a plane to convert.
    got = _of(dz_um, lock=5388.0, factors={OBJ: 10.0})
    assert got.outcome == outcome
    assert got.pfs_offset == offset
    if outcome == "derived":
        assert got.derived is not None
    else:
        assert got.pfs_offset is None
        assert got.derived is None
        assert got.z_um == Z_UM


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_distance_that_is_not_a_number_is_implausible(bad: float) -> None:
    # `model_construct` writes what validation would refuse: a run document is
    # a file, and the rule does not trust what read it.
    focus = FieldFocus.model_construct(
        source="well", sweep=None, dz_um=bad, frame_score=0.3, duration_s=0.25
    )
    got = stop_focus(
        _frame(focus), objective=OBJ, factors=FACTORS, offset_limits=LIMITS
    )
    assert got.outcome == "implausible"
    assert got.pfs_offset is None
    assert got.z_um == Z_UM


def test_a_distance_is_judged_before_the_objective_is_looked_up() -> None:
    # §26.3's order: step 4 (the distance) before step 5 (the objective).
    got = _of(50.1, objective=None)
    assert got.outcome == "implausible"


@pytest.mark.parametrize(
    ("focus", "pfs_offset"), [("unlocked", 4000.0), ("off", None), ("none", None)]
)
@pytest.mark.parametrize("bad", [60.0, -60.0, math.nan, math.inf])
def test_a_frame_that_is_not_derived_is_recorded_whatever_its_distance(
    focus: FocusState, pfs_offset: float | None, bad: float
) -> None:
    # §26.3's order: step 3 (nothing to convert) before step 4 (the distance). An
    # unlocked, `off` or `none` frame is left as the scan recorded it, and a
    # distance that would be refused on a locked frame is not looked at.
    cell_focus = FieldFocus.model_construct(
        source="well", sweep=None, dz_um=bad, frame_score=0.3, duration_s=0.25
    )
    got = stop_focus(
        _frame(cell_focus, focus=focus, pfs_offset=pfs_offset),
        objective=OBJ,
        factors=FACTORS,
        offset_limits=LIMITS,
    )
    assert (got.outcome, got.z_um, got.pfs_offset) == ("recorded", Z_UM, pfs_offset)
    assert got.derived is None


# --- the plane that was not found -----------------------------------------------------


@pytest.mark.parametrize("outcome", ["flat", "at-edge", "two-peaks"])
def test_a_sweep_without_a_peak_is_flagged_and_left_as_recorded(outcome: str) -> None:
    # D8: a sweep that found no peak left the frame at its centre, so Z and
    # offset agree and name the plane the autofocus held, not the cells'. A
    # number cannot say that the plane was not found: the stop is counted.
    focus = _focus(0.0, source="swept", outcome=outcome)
    got = stop_focus(
        _frame(focus), objective=OBJ, factors=FACTORS, offset_limits=LIMITS
    )
    assert (got.outcome, got.pfs_offset, got.z_um) == ("recorded", 5388.0, Z_UM)
    assert got.no_peak is True


@pytest.mark.parametrize(
    ("dz_um", "objective", "factors", "outcome"),
    [
        (-3.0, OBJ, FACTORS, "derived"),
        (60.0, OBJ, FACTORS, "implausible"),
        (-3.0, None, FACTORS, "no-objective"),
        (-3.0, "Nikon 10X S Fluor", FACTORS, "no-factor"),
        (50.0, OBJ, {OBJ: 1e308}, "no-factor"),
        (-3.0, OBJ, {OBJ: 1e9}, "outside-limits"),
    ],
)
def test_a_sweep_without_a_peak_is_flagged_whatever_became_of_the_offset(
    dz_um: float,
    objective: str | None,
    factors: dict[str, float],
    outcome: StopOutcome,
) -> None:
    # §26.3 step 2 does not depend on the offset: a record whose sweep found
    # nothing and that still has a distance (edited, or written by a scan that
    # keeps one) is counted in the no-peak line besides being derived or refused.
    got = stop_focus(
        _frame(_focus(dz_um, source="swept", outcome="flat")),
        objective=objective,
        factors=factors,
        offset_limits=LIMITS,
    )
    assert got.outcome == outcome
    assert got.no_peak is True


def test_a_peak_and_a_well_that_reused_a_distance_are_not_flagged() -> None:
    peak = stop_focus(
        _frame(_focus(0.0, source="swept", outcome="peak")),
        objective=OBJ,
        factors=FACTORS,
        offset_limits=LIMITS,
    )
    reused = stop_focus(
        _frame(_focus(0.0, source="well")),
        objective=OBJ,
        factors=FACTORS,
        offset_limits=LIMITS,
    )
    assert peak.no_peak is False
    assert reused.no_peak is False


def test_a_frame_with_no_peak_is_flagged_whatever_its_focus_state() -> None:
    # The flag is about the sweep, not about the lock: an unlocked frame whose
    # sweep found nothing is counted too.
    got = stop_focus(
        _frame(
            _focus(0.0, source="swept", outcome="flat"),
            focus="unlocked",
            pfs_offset=4000.0,
        ),
        objective=OBJ,
        factors=FACTORS,
        offset_limits=LIMITS,
    )
    assert (got.outcome, got.no_peak) == ("recorded", True)


@pytest.mark.parametrize(
    "frame_args",
    [
        {"cell_focus": None},
        {"cell_focus": _focus(0.0)},
        {"cell_focus": _focus(-3.0)},
        {"cell_focus": _focus(-3.0), "focus": "unlocked"},
        {"cell_focus": _focus(60.0)},
    ],
    ids=["no-step", "centre", "derived", "unlocked", "implausible"],
)
@pytest.mark.parametrize("objective", [None, OBJ])
def test_z_is_the_frames_in_every_outcome(
    frame_args: dict[str, Any], objective: str | None
) -> None:
    # D7: `z_um` is the readback at the cells' plane, whatever happens to the
    # offset beside it.
    for z_um in (Z_UM, None):
        got = stop_focus(
            _frame(z_um=z_um, **frame_args),
            objective=objective,
            factors=FACTORS,
            offset_limits=(0.0, 100.0),
        )
        assert got.z_um == z_um


# --- the texts ---------------------------------------------------------------------


def _one(
    outcome: StopOutcome,
    *,
    dz_um: float = -3.0,
    factor: float = 112.6,
    refused: float | None = None,
    no_peak: bool = False,
) -> StopFocus:
    derived = (
        OffsetFromFocus(lock_offset=5388.0, dz_um=dz_um, offset_per_um=factor)
        if outcome == "derived"
        else None
    )
    return StopFocus(
        z_um=Z_UM,
        pfs_offset=5050.0 if outcome in ("derived", "recorded") else None,
        derived=derived,
        outcome=outcome,
        refused_offset=refused,
        no_peak=no_peak,
    )


def _lines(
    stops: list[StopFocus],
    *,
    total: int = 20,
    objective: str | None = OBJ,
    limits: tuple[float, float] | None = LIMITS,
) -> tuple[str, ...]:
    return focus_assumed(stops, total=total, objective=objective, offset_limits=limits)


def test_each_line_is_the_one_of_the_design() -> None:
    # §26.6, character for character.
    assert _lines([_one("derived", dz_um=-3.0), _one("derived", dz_um=2.7)]) == (
        "2 of 20 stops carry a pfs_offset derived for the cells' plane, not read "
        "with PFS locked there: their frame was taken -3.00…+2.70 µm from the "
        "plane PFS held, and the offset is the lock's plus that distance times "
        "the profile's factor (one number for a factor that varies with the "
        "offset), rounded to a whole unit; 112.6 offset units per µm for "
        "Nikon 40X Plan Fluor ELWD",
    )
    assert _lines([_one("no-objective")] * 3) == (
        "3 of 20 stops have no pfs_offset: their frame was taken away from the "
        "plane PFS held (the focus on the cells) and the scan recorded no "
        "objective, so the profile's offset_per_um factor cannot be looked up; "
        "they keep z_um, the cells' plane",
    )
    assert _lines([_one("no-factor")] * 4) == (
        "4 of 20 stops have no pfs_offset: their frame was taken away from the "
        "plane PFS held (the focus on the cells) and the profile has no "
        "[autofocus.offset_per_um] factor for the scan's objective; they keep "
        "z_um, the cells' plane (objective: Nikon 40X Plan Fluor ELWD)",
    )
    assert _lines(
        [_one("outside-limits", refused=20001.0), _one("outside-limits", refused=-1.0)]
    ) == (
        "2 of 20 stops have no pfs_offset: the offset derived for the cells' "
        "plane (20001 for the first) lies outside the profile's "
        "autofocus_offset_soft_limits [0, 20000]; they keep z_um, the cells' plane",
    )
    assert _lines([_one("implausible")] * 5) == (
        "5 of 20 stops have no pfs_offset: their frame's plane is not within 50 "
        "µm of the plane PFS held (the focus sweep's largest range), so the "
        "record cannot be right; run scan again; they keep z_um",
    )
    assert _lines([_one("recorded", no_peak=True)] * 6) == (
        "6 of 20 stops took their frame from a field whose focus sweep found no "
        "peak (flat, at an edge or two peaks): its plane is the sweep's centre, "
        "not the cells' plane",
    )


def test_the_lines_come_in_the_designs_order_each_only_when_there_is_one() -> None:
    stops = [
        _one("recorded", no_peak=True),
        _one("implausible"),
        _one("outside-limits", refused=-1.0),
        _one("no-factor"),
        _one("no-objective"),
        _one("derived"),
        _one("recorded"),
    ]
    lines = _lines(stops, total=7)
    markers = (
        "carry a pfs_offset derived",
        "the scan recorded no objective",
        "[autofocus.offset_per_um]",
        "autofocus_offset_soft_limits",
        "run scan again",
        "found no peak",
    )
    assert len(lines) == len(markers)
    for line, marker in zip(lines, markers, strict=True):
        assert marker in line
        assert line.startswith("1 of 7 stops ")


def test_a_route_without_a_focus_step_gets_no_line() -> None:
    recorded = [_one("recorded")] * 5
    assert _lines(recorded, total=5) == ()
    assert _lines([], total=0) == ()
    # And a stop that the route had no frame for (another objective): nothing.
    none_stop = stop_focus(None, objective=OBJ, factors=FACTORS, offset_limits=None)
    assert _lines([none_stop] * 3, total=3) == ()


def test_the_derived_range_is_the_smallest_and_largest_distance() -> None:
    # Of the derived stops only: a refused or a recorded stop's distance is not
    # in it.
    stops = [
        _one("derived", dz_um=1.234),
        _one("derived", dz_um=-4.567),
        _one("derived", dz_um=0.5),
        _one("implausible", dz_um=99.0),
        _one("recorded"),
    ]
    line, _implausible = _lines(stops, total=5)
    assert "-4.57…+1.23 µm" in line
    assert line.startswith("3 of 5 stops carry")
    (same,) = _lines([_one("derived", dz_um=2.0), _one("derived", dz_um=2.0)])
    assert "+2.00…+2.00 µm" in same


def test_the_factors_sign_and_the_limits_are_said_as_they_are() -> None:
    (line,) = _lines([_one("derived", factor=-112.6)])
    assert "; -112.6 offset units per µm for " in line
    # `{:g}`: a whole-number factor has no ".0" (the design's `100`, not `100.0`).
    (whole,) = _lines([_one("derived", factor=100.0)])
    assert "; 100 offset units per µm for " in whole
    (limits,) = _lines([_one("outside-limits", refused=-0.4)], limits=(-5.5, 20000.0))
    assert "(-0 for the first)" in limits
    assert "[-5.5, 20000]" in limits


_LABEL = "Nikon 40X Plan Fluor ELWD " + "x" * 34  # 60 characters


def _longest_forms(label: str) -> dict[str, str]:
    """Every line of the design in the longest form it takes, by its outcome."""
    derived = [
        _one("derived", dz_um=-50.0, factor=-112.6),
        _one("derived", dz_um=50.0, factor=-112.6),
        *[_one("derived", dz_um=0.5, factor=-112.6)] * 118,
    ]
    many = 120
    return {
        "derived": _lines(derived, total=many, objective=label)[0],
        "no-objective": _lines([_one("no-objective")] * many, total=many)[0],
        "no-factor": _lines([_one("no-factor")] * many, total=many, objective=label)[0],
        "outside-limits": _lines(
            [_one("outside-limits", refused=-20001.0)] * many, total=many
        )[0],
        "implausible": _lines([_one("implausible")] * many, total=many)[0],
        "no-peak": _lines([_one("recorded", no_peak=True)] * many, total=many)[0],
    }


def test_the_longest_forms_of_the_lines_stay_under_500_characters() -> None:
    # §26.6, measured against the 600 characters the plate page's card keeps
    # (FM-205): the longest form of each line (a 60-character label,
    # `-50.00…+50.00`, a factor of `-112.6`, limits `[0, 20000]`, 120 stops), not
    # the demo's.
    forms = _longest_forms(_LABEL)
    assert set(forms) == {
        "derived",
        "no-objective",
        "no-factor",
        "outside-limits",
        "implausible",
        "no-peak",
    }
    assert {name: len(line) for name, line in forms.items() if len(line) > 500} == {}
    assert "-50.00…+50.00" in forms["derived"]
    assert "-112.6 offset units per µm" in forms["derived"]
    assert "[0, 20000]" in forms["outside-limits"]
    assert forms["derived"].endswith(_LABEL)
    assert forms["no-factor"].endswith(_LABEL + ")")


def test_the_lines_keep_their_words_before_the_objective_and_stay_short() -> None:
    # FM-205, FM-139: the card cuts a line at 600 characters, so the words that
    # matter come first and the free text, the objective's label, is last (the
    # no-factor line closes its parenthesis after it). A 700-character label
    # with a line break must leave everything before it in one line, whole.
    label = "A\nB" + "x" * 697
    forms = _longest_forms(label)
    for name, tail in (("derived", label), ("no-factor", label + ")")):
        line = forms[name]
        assert line.endswith(tail)
        fixed = line[: -len(label) - (len(tail) - len(label))]
        assert "\n" not in fixed
        assert len(fixed) <= 500
        # What the card does: one line, then the first 600 characters.
        flat = " ".join(line.splitlines())
        assert flat[:600].startswith(fixed)
    for name in ("no-objective", "outside-limits", "implausible", "no-peak"):
        assert "\n" not in forms[name]
        assert len(forms[name]) <= 500


def test_the_plate_pages_card_keeps_every_line_whole_before_a_long_label() -> None:
    # FM-205: the longest form of each line goes through the card builder itself,
    # not through a model of it, so a change of its bound or of its `_cut(_flat())`
    # shows here. A 700-character label with a line break is the free text.
    from datetime import datetime, timezone

    from smc.run import StageRecord, StageStatus
    from smc.ui.controller import _card

    label = "A\nB" + "x" * 697
    forms = _longest_forms(label)
    then = datetime(2026, 11, 13, 9, 30, tzinfo=timezone.utc)
    record = StageRecord(
        plugin="route",
        provides="route",
        attempt="a1",
        status=StageStatus.done,
        started_utc=then,
        finished_utc=then,
        profile="demo",
        dry_run=False,
        smc_version="0",
        params={},
        result={"assumed": list(forms.values())},
    )

    shown = _card(record).assumed

    assert len(shown) == len(forms)  # the card keeps eight lines at most
    for line, kept in zip(forms.values(), shown, strict=True):
        assert "\n" not in kept
        fixed = line[: line.index(label)] if label in line else line
        # Everything the line says before the label is there, and for a line with
        # no label, the whole line: no "…" where a word was.
        assert kept.startswith(fixed)
        if label not in line:
            assert kept == line
