"""The measurement of a detector against a reference: matching, confusion, scores (m3 §9.4, #342).

Pure arithmetic on boxes, so every test builds its boxes by hand and the expected
numbers are written from them, not from the code. Boxes are ``(row0, col0, row1,
col1)``; the ones here are not square, because a box that exchanged its two axes
would score the same on a square one (FM-73).
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.cells.evaluate import (
    DEFAULT_MATCH_IOU,
    NO_BOX,
    ClassScore,
    LabelledBox,
    class_scores,
    confusion,
    match_boxes,
)

CLASSES = ("single", "doublet", "debris")


def _box(label: str, row0: float, col0: float, row1: float, col1: float) -> LabelledBox:
    return LabelledBox((row0, col0, row1, col1), label)


# --- match_boxes ---------------------------------------------------------------


def test_a_detection_matches_one_reference_box_only() -> None:
    """Two reference boxes over one detection are one hit and one miss, not two hits."""
    reference = [(0.0, 0.0, 10.0, 20.0), (0.0, 2.0, 10.0, 22.0)]
    detected = [(0.0, 1.0, 10.0, 21.0)]

    pairs = match_boxes(reference, detected)

    # Both reference boxes fit it as well (IoU 0.9): the lower index keeps it.
    assert pairs == [(0, 0)]


def test_a_reference_box_is_matched_by_one_detection_only() -> None:
    reference = [(0.0, 0.0, 10.0, 20.0)]
    detected = [(0.0, 0.0, 10.0, 20.0), (0.0, 1.0, 10.0, 21.0)]

    assert match_boxes(reference, detected) == [(0, 0)]


def test_the_highest_iou_pair_is_matched_first() -> None:
    """Reference 0 would take detection 0 in the order of the lists; the better pair wins it."""
    reference = [(0.0, 0.0, 10.0, 10.0), (0.0, 2.0, 10.0, 12.0)]
    # IoU with reference 0: 80 / 120 = 0.67 (a candidate); with reference 1: 1.0.
    detected = [(0.0, 2.0, 10.0, 12.0)]

    assert match_boxes(reference, detected) == [(1, 0)]


def test_the_runner_up_takes_the_box_the_best_pair_left() -> None:
    """After (ref 1, det 0) is taken, ref 0 matches det 1 on its own merit."""
    reference = [(0.0, 0.0, 10.0, 10.0), (0.0, 2.0, 10.0, 12.0)]
    # det 0: 1.0 with ref 1, 0.67 with ref 0. det 1: 0.9 with ref 0, 0.58 with ref 1.
    detected = [(0.0, 2.0, 10.0, 12.0), (0.0, 0.0, 10.0, 9.0)]

    assert match_boxes(reference, detected) == [(0, 1), (1, 0)]


def test_equal_ious_are_taken_by_reference_index_then_detection_index() -> None:
    """Two detections that fit one reference box equally: the first of them, and the answer is stable."""
    reference = [(0.0, 0.0, 10.0, 10.0)]
    detected = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)]

    assert match_boxes(reference, detected) == [(0, 0)]


def test_a_pair_below_the_iou_is_not_a_match() -> None:
    reference = [(0.0, 0.0, 10.0, 10.0)]

    # 45 / 100 = 0.45 against 0.5, and 55 / 100 = 0.55.
    assert match_boxes(reference, [(0.0, 0.0, 10.0, 4.5)]) == []
    assert match_boxes(reference, [(0.0, 0.0, 10.0, 5.5)]) == [(0, 0)]


def test_a_pair_at_exactly_the_iou_is_a_match() -> None:
    assert DEFAULT_MATCH_IOU == 0.5
    reference = [(0.0, 0.0, 10.0, 10.0)]

    assert match_boxes(reference, [(0.0, 0.0, 10.0, 5.0)]) == [(0, 0)]


def test_the_iou_asked_for_is_the_one_used() -> None:
    reference = [(0.0, 0.0, 10.0, 10.0)]
    detected = [(0.0, 0.0, 10.0, 7.0)]

    assert match_boxes(reference, detected, min_iou=0.6) == [(0, 0)]
    assert match_boxes(reference, detected, min_iou=0.8) == []


def test_boxes_that_only_touch_or_lie_apart_do_not_match() -> None:
    reference = [(0.0, 0.0, 10.0, 10.0)]

    assert match_boxes(reference, [(0.0, 10.0, 10.0, 20.0)]) == []
    assert match_boxes(reference, [(50.0, 50.0, 60.0, 60.0)]) == []


def test_a_box_is_read_rows_first() -> None:
    """A tall box and a wide box of the same area overlap by a thin strip, whichever way it is read (FM-73)."""
    tall = (0.0, 0.0, 20.0, 5.0)
    wide = (0.0, 0.0, 5.0, 20.0)
    # Intersection 25, union 175: IoU 0.143. Swapping axes of one box only would make
    # them identical (IoU 1.0): the two must not match.
    assert match_boxes([tall], [wide]) == []
    assert match_boxes([tall], [tall]) == [(0, 0)]


def test_no_box_on_either_side_is_no_pair() -> None:
    assert match_boxes([], []) == []
    assert match_boxes([(0.0, 0.0, 10.0, 10.0)], []) == []
    assert match_boxes([], [(0.0, 0.0, 10.0, 10.0)]) == []


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, math.nan, math.inf])
def test_a_match_iou_outside_zero_one_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="min_iou"):
        match_boxes([(0.0, 0.0, 1.0, 1.0)], [(0.0, 0.0, 1.0, 1.0)], min_iou=bad)


def test_a_match_iou_of_one_asks_for_the_same_box() -> None:
    box = (0.0, 0.0, 10.0, 10.0)

    assert match_boxes([box], [box], min_iou=1.0) == [(0, 0)]
    assert match_boxes([box], [(0.0, 0.0, 10.0, 9.9)], min_iou=1.0) == []


_COORD = st.integers(0, 40)


@st.composite
def _boxes(draw: st.DrawFn) -> tuple[float, float, float, float]:
    row0, col0 = draw(_COORD), draw(_COORD)
    return (
        float(row0),
        float(col0),
        float(row0 + draw(st.integers(1, 20))),
        float(col0 + draw(st.integers(1, 20))),
    )


@given(st.lists(_boxes(), max_size=8), st.lists(_boxes(), max_size=8))
def test_a_matching_is_one_to_one_whatever_the_boxes(
    reference: list[tuple[float, float, float, float]],
    detected: list[tuple[float, float, float, float]],
) -> None:
    pairs = match_boxes(reference, detected)

    assert len({r for r, _ in pairs}) == len(pairs)
    assert len({d for _, d in pairs}) == len(pairs)
    assert len(pairs) <= min(len(reference), len(detected))


# --- confusion -----------------------------------------------------------------


def test_a_matched_pair_is_counted_under_both_labels() -> None:
    reference = [_box("single", 0, 0, 10, 20)]
    detected = [_box("single", 0, 1, 10, 21)]

    assert confusion(reference, detected) == Counter({("single", "single"): 1})


def test_a_box_nobody_matched_is_a_miss_or_a_false_box() -> None:
    reference = [_box("single", 0, 0, 10, 20), _box("debris", 50, 50, 60, 70)]
    detected = [_box("single", 0, 1, 10, 21), _box("doublet", 100, 100, 110, 120)]

    assert confusion(reference, detected) == Counter(
        {
            ("single", "single"): 1,
            ("debris", NO_BOX): 1,
            (NO_BOX, "doublet"): 1,
        }
    )


def test_a_single_found_as_a_doublet_is_a_match_with_the_wrong_label() -> None:
    """Not a miss and a false box: the cell was seen, and named wrongly."""
    reference = [_box("single", 0, 0, 10, 20)]
    detected = [_box("doublet", 0, 0, 10, 20)]

    assert confusion(reference, detected) == Counter({("single", "doublet"): 1})


def test_the_counters_of_two_frames_add_up() -> None:
    first = confusion([_box("single", 0, 0, 10, 20)], [_box("single", 0, 0, 10, 20)])
    second = confusion([_box("single", 0, 0, 10, 20)], [])

    assert first + second == Counter({("single", "single"): 1, ("single", NO_BOX): 1})


def test_an_empty_frame_gives_an_empty_confusion() -> None:
    assert confusion([], []) == Counter()


def test_the_confusion_takes_the_iou_it_is_given() -> None:
    reference = [_box("single", 0, 0, 10, 10)]
    detected = [_box("single", 0, 0, 10, 4.5)]

    assert confusion(reference, detected) == Counter(
        {("single", NO_BOX): 1, (NO_BOX, "single"): 1}
    )
    assert confusion(reference, detected, min_iou=0.4) == Counter(
        {("single", "single"): 1}
    )


# --- class_scores --------------------------------------------------------------


def test_a_wrong_label_counts_against_both_classes() -> None:
    """The single found as a doublet is a missed single and a false doublet."""
    counts = Counter({("single", "doublet"): 1, ("doublet", "doublet"): 1})

    per_class, any_box = class_scores(counts, CLASSES)

    assert per_class["single"] == ClassScore(
        reference=1, detected=0, matched=0, precision=None, recall=0.0
    )
    assert per_class["doublet"] == ClassScore(
        reference=1, detected=2, matched=1, precision=0.5, recall=1.0
    )
    assert any_box == ClassScore(
        reference=2, detected=2, matched=2, precision=1.0, recall=1.0
    )


def test_a_class_never_detected_has_no_precision() -> None:
    """``None``, not 0 and not nan: a class the detector never gives has no precision to speak of."""
    counts = Counter({("debris", NO_BOX): 3, ("single", "single"): 2})

    per_class, _ = class_scores(counts, CLASSES)

    assert per_class["debris"].precision is None
    assert per_class["debris"].recall == 0.0
    assert per_class["debris"].detected == 0


def test_a_class_never_in_the_reference_has_no_recall() -> None:
    counts = Counter({(NO_BOX, "debris"): 2})

    per_class, _ = class_scores(counts, CLASSES)

    assert per_class["debris"].recall is None
    assert per_class["debris"].precision == 0.0


def test_a_class_in_neither_has_neither_score() -> None:
    per_class, any_box = class_scores(Counter(), CLASSES)

    assert per_class["doublet"] == ClassScore(0, 0, 0, None, None)
    assert any_box == ClassScore(0, 0, 0, None, None)


def test_any_box_ignores_the_labels() -> None:
    counts = Counter(
        {
            ("single", "doublet"): 1,  # a match, wrong label: counts for any box
            ("single", "single"): 1,
            ("debris", NO_BOX): 1,
            (NO_BOX, "single"): 2,
        }
    )

    _, any_box = class_scores(counts, CLASSES)

    assert any_box == ClassScore(
        reference=3, detected=4, matched=2, precision=0.5, recall=pytest.approx(2 / 3)
    )


def test_the_scores_count_every_pair_that_names_the_class() -> None:
    counts = Counter(
        {
            ("single", "single"): 4,
            ("single", NO_BOX): 1,
            (NO_BOX, "single"): 3,
            ("doublet", "single"): 2,
        }
    )

    per_class, _ = class_scores(counts, CLASSES)

    # reference: (single, .) = 4 + 1; detected: (., single) = 4 + 3 + 2.
    assert per_class["single"] == ClassScore(
        reference=5,
        detected=9,
        matched=4,
        precision=pytest.approx(4 / 9),
        recall=0.8,
    )


def test_a_label_outside_the_classes_counts_for_any_box_only() -> None:
    counts = Counter({("cell", "cell"): 1, ("single", "single"): 1})

    per_class, any_box = class_scores(counts, CLASSES)

    assert per_class["single"].reference == 1
    assert any_box.reference == 2
    assert any_box.matched == 2


def test_a_class_named_like_the_no_box_marker_is_refused() -> None:
    with pytest.raises(ValueError, match=NO_BOX):
        class_scores(Counter(), ("single", NO_BOX))


def test_the_scores_come_back_for_every_class_asked_in_order() -> None:
    per_class, _ = class_scores(Counter(), CLASSES)

    assert tuple(per_class) == CLASSES
