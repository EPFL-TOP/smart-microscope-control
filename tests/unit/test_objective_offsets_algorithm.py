"""The arithmetic of ``objective-offsets`` (m3 §20.5), on numbers written out by hand.

Nothing here reads a device or a file. The expected values are the ones a
person gets with a pencil from the shifts below, never a call of the function
under test (FM-124): the reference's own shift is not zero in the case that
moves the offsets to another objective (FM-203), the shifts are anisotropic and
far above any tolerance (FM-73), and the numbers are the worked example of m2
§13.2 and m3 §18: 10X at ``(5.0, -2.5)``, 40X at ``(-32.5, 9.75)`` from the
20X, so that the shift 10X -> 40X is ``(-37.5, 12.25)``.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.objective_offsets.algorithm import (
    MAX_TARGETS,
    SPREAD_PIXELS,
    default_max_spread_um,
    mean_shift_um,
    offsets_from_reference,
    spread_um,
)

FOUR = "Nikon 4X Plan Fluor"
TEN = "Nikon 10X S Fluor"
FORTY = "Nikon 40X Plan Fluor ELWD"


def test_the_bounds_are_the_ones_the_design_chose() -> None:
    assert SPREAD_PIXELS == 2.0
    assert MAX_TARGETS == 5


def test_the_shift_is_the_mean_of_the_wells_differences() -> None:
    deltas = [(1.0, 2.0), (3.0, 6.0), (5.0, 10.0)]

    assert mean_shift_um(deltas) == (3.0, 6.0)


def test_the_mean_of_one_difference_is_that_difference() -> None:
    assert mean_shift_um([(-37.5, 12.25)]) == (-37.5, 12.25)


def test_the_mean_is_per_axis_and_keeps_the_sign() -> None:
    # An x that is always positive and a y that is always negative: a mean of
    # the lengths, or of absolute values, would lose both signs.
    assert mean_shift_um([(4.0, -2.0), (6.0, -4.0)]) == (5.0, -3.0)


def test_the_spread_is_the_largest_distance_between_two_wells() -> None:
    # The pairs are 1 (a, b), 10 (a, c) and sqrt(101) (b, c): the last one.
    deltas = [(0.0, 0.0), (1.0, 0.0), (0.0, 10.0)]

    assert spread_um(deltas) == pytest.approx(math.sqrt(101.0), abs=1e-12)


def test_the_spread_of_a_three_four_five_pair_is_five() -> None:
    assert spread_um([(0.0, 0.0), (3.0, 4.0)]) == 5.0


def test_one_well_has_no_spread() -> None:
    assert spread_um([(-37.5, 12.25)]) == 0.0


def test_the_spread_does_not_depend_on_where_the_wells_differences_lie() -> None:
    # A translation of every difference is the same shift of the target, and
    # what disagrees is what the spread measures.
    deltas = [(0.0, 0.0), (3.0, 4.0)]
    moved = [(100.0, -50.0), (103.0, -46.0)]

    assert spread_um(moved) == spread_um(deltas) == 5.0


@pytest.mark.parametrize("function", [mean_shift_um, spread_um])
def test_no_well_is_refused(function: object) -> None:
    with pytest.raises(ValueError, match="no difference"):
        function([])  # type: ignore[operator]


def test_the_default_bound_is_two_pixels_of_the_coarser_objective() -> None:
    # The 4X's 2.75 µm against the 10X's 1.1 µm: two of the 4X's, 5.5 µm; and
    # the other way round, the same.
    assert default_max_spread_um(2.75, 1.1) == 5.5
    assert default_max_spread_um(1.1, 2.75) == 5.5
    assert default_max_spread_um(1.0, 1.0) == 2.0


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_a_pixel_size_that_is_not_finite_and_positive_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="pixel size"):
        default_max_spread_um(bad, 1.0)
    with pytest.raises(ValueError, match="pixel size"):
        default_max_spread_um(1.0, bad)


def test_offsets_are_the_shifts_when_the_profile_names_no_reference() -> None:
    reference, offsets = offsets_from_reference(
        FOUR, [(FORTY, (-37.5, 12.25)), (TEN, (-5.0, 2.5))], None
    )

    assert reference == FOUR
    assert offsets == ((FORTY, (-37.5, 12.25)), (TEN, (-5.0, 2.5)))


def test_the_reference_may_be_the_registrations_objective_itself() -> None:
    reference, offsets = offsets_from_reference(FOUR, [(TEN, (-5.0, 2.5))], FOUR)

    assert reference == FOUR
    assert offsets == ((TEN, (-5.0, 2.5)),)


def test_offsets_are_moved_to_a_reference_among_the_targets() -> None:
    # The registration's objective "A" and two targets: B shifted by (10, -4)
    # and C by (-6, 8), both from A. With B as the reference, the offset of an
    # objective is what its feature is moved by from B's: A's is -(10, -4),
    # C's is (-6, 8) - (10, -4).
    reference, offsets = offsets_from_reference(
        "A", [("B", (10.0, -4.0)), ("C", (-6.0, 8.0))], "B"
    )

    assert reference == "B"
    assert offsets == (("A", (-10.0, 4.0)), ("C", (-16.0, 12.0)))


def test_the_worked_example_of_the_design_with_the_20x_as_the_reference() -> None:
    # m2 §13.2: from the 20X the 10X is at (5.0, -2.5) and the 40X at (-32.5,
    # 9.75). A run registered at the 10X measured the 20X at (-5.0, 2.5) and the
    # 40X at (-37.5, 12.25) from it; with the 20X as the reference the profile's
    # numbers come back.
    twenty = "Nikon 20X Plan Fluor ELWD"
    reference, offsets = offsets_from_reference(
        TEN, [(twenty, (-5.0, 2.5)), (FORTY, (-37.5, 12.25))], twenty
    )

    assert reference == twenty
    assert offsets == ((TEN, (5.0, -2.5)), (FORTY, (-32.5, 9.75)))


def test_the_registrations_objective_comes_first_then_the_targets_in_order() -> None:
    _, offsets = offsets_from_reference(
        "A", [("D", (1.0, 1.0)), ("B", (2.0, 2.0)), ("C", (3.0, 3.0))], "B"
    )

    assert [label for label, _ in offsets] == ["A", "D", "C"]


def test_a_reference_the_run_did_not_measure_is_refused() -> None:
    with pytest.raises(ValueError, match="'Z'"):
        offsets_from_reference("A", [("B", (1.0, 1.0))], "Z")


def test_an_objective_given_twice_is_refused() -> None:
    with pytest.raises(ValueError, match=r"objective 'B' is given twice"):
        offsets_from_reference("A", [("B", (1.0, 1.0)), ("B", (2.0, 2.0))], None)
    with pytest.raises(ValueError, match=r"objective 'A' is given twice"):
        offsets_from_reference("A", [("A", (1.0, 1.0))], None)


def test_no_offset_is_a_negative_zero() -> None:
    # `-0.0` prints as "-0.0" in a profile and in a line the operator reads.
    _, offsets = offsets_from_reference("A", [("B", (-0.0, 0.0))], None)
    _, moved = offsets_from_reference("A", [("B", (0.0, 0.0)), ("C", (5.0, 0.0))], "B")

    for _, xy in (*offsets, *moved):
        assert [math.copysign(1.0, v) for v in xy] == [1.0, 1.0]


_SHIFT = st.tuples(
    st.floats(min_value=-1000.0, max_value=1000.0),
    st.floats(min_value=-1000.0, max_value=1000.0),
)


@given(a=_SHIFT, b=_SHIFT, c=_SHIFT)
def test_changing_the_reference_is_a_translation_of_every_offset(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> None:
    # The offsets from B and from C differ, for each objective, by the
    # same amount: the position of C seen from B. So the offsets of m2 §13.2
    # stay one coherent frame whichever objective is the reference.
    shifts = [("B", a), ("C", b), ("D", c)]
    _, from_b = offsets_from_reference("A", shifts, "B")
    _, from_c = offsets_from_reference("A", shifts, "C")
    by_b = dict(from_b)
    by_c = dict(from_c)
    # What C's offset from B is, and B's from C: opposite.
    assert by_b["C"][0] == pytest.approx(-by_c["B"][0], abs=1e-9)
    assert by_b["C"][1] == pytest.approx(-by_c["B"][1], abs=1e-9)
    # A third objective's offsets differ by exactly C's from B.
    for label in ("A", "D"):
        assert by_b[label][0] - by_c[label][0] == pytest.approx(by_b["C"][0], abs=1e-9)
        assert by_b[label][1] - by_c[label][1] == pytest.approx(by_b["C"][1], abs=1e-9)
