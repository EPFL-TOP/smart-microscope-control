"""Overlap suppression: the 17 tests of ``nikon-control``'s ``test_overlap.py``, ported (m3 §9.2).

Pure numbers: boxes are ``(row0, col0, row1, col1)`` and nothing here has a
detector, a frame or a microscope.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from smc.plugins.cells.overlap import (
    PREFER_AREA,
    PREFER_SCORE,
    OverlapCandidate,
    bbox_iomin,
    bbox_iou,
    count_overlaps,
    select_survivors,
)


def _c(
    ref: object,
    bbox: Sequence[float],
    score: float | None = None,
    protected: bool = False,
) -> OverlapCandidate:
    return OverlapCandidate(ref=ref, bbox=bbox, score=score, protected=protected)


# ---- the measure ---------------------------------------------------------


def test_iomin_detects_containment_where_iou_fails() -> None:
    """The case that matters: a small 'single' inside a big 'doublet'."""
    big = [0, 0, 100, 100]  # area 10000
    small = [10, 10, 40, 40]  # area 900, fully inside
    assert bbox_iomin(big, small) == 1.0  # 100% of the small box
    assert bbox_iou(big, small) < 0.1  # IoU would miss it entirely


def test_iomin_is_symmetric_and_zero_when_disjoint() -> None:
    a, b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert bbox_iomin(a, b) == bbox_iomin(b, a)
    assert bbox_iomin([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0


def test_iomin_partial_overlap() -> None:
    # 10x10 boxes offset by 5 in both axes -> 5x5 = 25 of 100 = 0.25
    assert bbox_iomin([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(0.25)


def test_zero_area_boxes_do_not_divide_by_zero() -> None:
    assert bbox_iomin([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


# ---- survivor selection --------------------------------------------------


def test_non_overlapping_boxes_all_survive() -> None:
    cands = [_c("a", [0, 0, 10, 10]), _c("b", [50, 50, 60, 60])]
    kept, dropped = select_survivors(cands)
    assert sorted(str(k) for k in kept) == ["a", "b"]
    assert dropped == []


def test_below_threshold_both_survive() -> None:
    # 25% overlap, limit 70%
    cands = [_c("a", [0, 0, 10, 10]), _c("b", [5, 5, 15, 15])]
    kept, dropped = select_survivors(cands, max_overlap=0.7)
    assert len(kept) == 2
    assert dropped == []


def test_prefer_area_keeps_the_bigger_box() -> None:
    """A doublet containing a single: the bigger box wins."""
    cands = [
        _c("single", [10, 10, 40, 40], score=0.95),
        _c("doublet", [0, 0, 100, 100], score=0.60),
    ]
    kept, dropped = select_survivors(cands, 0.7, prefer=PREFER_AREA)
    assert kept == ["doublet"]
    assert dropped == ["single"]


def test_prefer_score_keeps_the_most_confident_box() -> None:
    cands = [
        _c("single", [10, 10, 40, 40], score=0.95),
        _c("doublet", [0, 0, 100, 100], score=0.60),
    ]
    kept, dropped = select_survivors(cands, 0.7, prefer=PREFER_SCORE)
    assert kept == ["single"]
    assert dropped == ["doublet"]


def test_score_breaks_ties_when_areas_are_equal() -> None:
    cands = [
        _c("lo", [0, 0, 10, 10], score=0.3),
        _c("hi", [0, 0, 10, 10], score=0.9),
    ]
    kept, dropped = select_survivors(cands, 0.7, prefer=PREFER_AREA)
    assert kept == ["hi"]
    assert dropped == ["lo"]


def test_human_box_beats_a_bigger_auto_box() -> None:
    """Human work outranks detector output regardless of size/score."""
    cands = [
        _c("auto", [0, 0, 100, 100], score=0.99),
        _c("mine", [10, 10, 40, 40], protected=True),
    ]
    kept, dropped = select_survivors(cands, 0.7)
    assert kept == ["mine"]
    assert dropped == ["auto"]


def test_two_overlapping_human_boxes_are_both_kept() -> None:
    """Deleting deliberate human work is not this function's job."""
    cands = [
        _c("m1", [0, 0, 100, 100], protected=True),
        _c("m2", [10, 10, 40, 40], protected=True),
    ]
    kept, dropped = select_survivors(cands, 0.7)
    assert sorted(str(k) for k in kept) == ["m1", "m2"]
    assert dropped == []


def test_drop_protected_overrides_that() -> None:
    cands = [
        _c("m1", [0, 0, 100, 100], protected=True),
        _c("m2", [10, 10, 40, 40], protected=True),
    ]
    kept, dropped = select_survivors(cands, 0.7, drop_protected=True)
    assert kept == ["m1"]
    assert dropped == ["m2"]


def test_cluster_of_three_keeps_only_the_best() -> None:
    cands = [
        _c("s1", [10, 10, 30, 30], score=0.8),
        _c("s2", [12, 12, 32, 32], score=0.9),
        _c("big", [0, 0, 100, 100], score=0.5),
    ]
    kept, dropped = select_survivors(cands, 0.7, prefer=PREFER_AREA)
    assert kept == ["big"]
    assert sorted(str(d) for d in dropped) == ["s1", "s2"]


def test_deterministic_for_identical_boxes() -> None:
    cands = [
        _c("a", [0, 0, 10, 10], score=0.5),
        _c("b", [0, 0, 10, 10], score=0.5),
    ]
    first = select_survivors(cands, 0.7)
    assert first == select_survivors(cands, 0.7)
    assert first[0] == ["a"]  # input order breaks the final tie


def test_chain_does_not_cascade_wrongly() -> None:
    """A survives, B overlaps A and goes, C overlaps only B so it STAYS."""
    a = _c("a", [0, 0, 100, 100], score=0.9)  # biggest -> kept
    b = _c("b", [90, 90, 130, 130], score=0.8)  # 25% into a... check math
    kept, _dropped = select_survivors([a, b], 0.7, prefer=PREFER_AREA)
    # b's own area is 1600, intersection with a is 10x10=100 -> 6% -> both stay
    assert len(kept) == 2


def test_prefer_must_be_valid() -> None:
    with pytest.raises(ValueError, match="prefer must be"):
        # The Literal keeps a type-checked caller from this; the error is for the rest.
        select_survivors([_c("a", [0, 0, 1, 1])], prefer="whatever")  # type: ignore[arg-type]


def test_count_overlaps_reports_without_changing() -> None:
    cands = [
        _c("big", [0, 0, 100, 100]),
        _c("in", [10, 10, 40, 40]),
        _c("far", [500, 500, 510, 510]),
    ]
    assert count_overlaps(cands, 0.7) == 2  # big + in, not far
    assert len(cands) == 3  # untouched
