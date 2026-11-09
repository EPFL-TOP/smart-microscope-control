"""How well a detector agrees with a reference set: matching, confusion, scores (m3 §9.4).

The model's counts are only as good as its detections on the stand's own frames,
and a reference set the owner annotated is how that is known. This is the
arithmetic of the comparison, numpy-free and torch-free: ``scripts/dev/cells_check.py``
runs a detector on the reference frames and hands the boxes here.

Boxes are ``(row0, col0, row1, col1)`` as ``Detection`` and ``Cell.box_px`` are.
COCO files, which the owner's review page writes, are columns first: the script
converts once at its edge (FM-73), so nothing here thinks about it.

The matching is ``nikon-control``'s: one-to-one, greedy by IoU, labels ignored.
A single the model calls a doublet is a **match with the wrong label**: the cell
was seen, and the confusion says it was named wrongly. Counting it as a miss and a
false box would punish one error twice and hide which class is confused with which.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from smc.plugins.cells.overlap import bbox_iou

__all__ = [
    "DEFAULT_MATCH_IOU",
    "NO_BOX",
    "ClassScore",
    "LabelledBox",
    "class_scores",
    "confusion",
    "match_boxes",
]

#: The label a box does not have on the side that missed it: ``(label, NO_BOX)`` is a
#: reference box nobody found, ``(NO_BOX, label)`` a detection nobody annotated.
NO_BOX = "none"
#: ``nikon-control``'s review page matches at this IoU too, so the two tools agree.
DEFAULT_MATCH_IOU = 0.5

_Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class LabelledBox:
    """A box in the pixels of its frame, ``(row0, col0, row1, col1)``, with its class name."""

    box_px: tuple[float, float, float, float]
    label: str


@dataclass(frozen=True, slots=True)
class ClassScore:
    """What was annotated, what was found, and how many of the two agree, for one class.

    ``precision`` is ``matched / detected`` and ``recall`` is ``matched / reference``.
    Each is ``None`` when its denominator is 0: a class the detector never gives
    (``debris`` for the blob detector) has no precision, which is not a precision of
    0, and a class nobody annotated has no recall.
    """

    reference: int
    detected: int
    matched: int
    precision: float | None
    recall: float | None


def match_boxes(
    reference: Sequence[_Box],
    detected: Sequence[_Box],
    *,
    min_iou: float = DEFAULT_MATCH_IOU,
) -> list[tuple[int, int]]:
    """Pairs ``(reference index, detection index)`` of boxes that are one object, labels ignored.

    Every pair whose IoU is at least ``min_iou`` is a candidate. They are taken by
    decreasing IoU, then by reference index, then by detection index, and a pair is
    kept when neither of its boxes is taken yet: one-to-one, and the same answer
    however the boxes were listed. The pairs come back sorted by reference index.

    Raises:
        ValueError: ``min_iou`` is not in ``(0, 1]`` (``nan`` included). Zero would
            match every pair of boxes that never touch.
    """
    if not 0.0 < min_iou <= 1.0:
        raise ValueError(f"min_iou must be in (0, 1], got {min_iou}")
    candidates = [
        (iou, r, d)
        for r, ref in enumerate(reference)
        for d, det in enumerate(detected)
        if (iou := bbox_iou(ref, det)) >= min_iou
    ]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    taken_reference: set[int] = set()
    taken_detection: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, r, d in candidates:
        if r in taken_reference or d in taken_detection:
            continue
        taken_reference.add(r)
        taken_detection.add(d)
        pairs.append((r, d))
    return sorted(pairs)


def confusion(
    reference: Sequence[LabelledBox],
    detected: Sequence[LabelledBox],
    *,
    min_iou: float = DEFAULT_MATCH_IOU,
) -> Counter[tuple[str, str]]:
    """Counts of ``(reference label, detected label)`` over one frame, misses included.

    A matched pair is counted under both labels, ``(label, NO_BOX)`` is a reference
    box that was not found and ``(NO_BOX, label)`` a detection that matches none.
    The counters of several frames add up with ``+``.

    Raises:
        ValueError: ``min_iou`` is not in ``(0, 1]``.
    """
    pairs = match_boxes(
        [box.box_px for box in reference],
        [box.box_px for box in detected],
        min_iou=min_iou,
    )
    counts: Counter[tuple[str, str]] = Counter()
    for r, d in pairs:
        counts[(reference[r].label, detected[d].label)] += 1
    found = {r for r, _ in pairs}
    used = {d for _, d in pairs}
    for r, box in enumerate(reference):
        if r not in found:
            counts[(box.label, NO_BOX)] += 1
    for d, box in enumerate(detected):
        if d not in used:
            counts[(NO_BOX, box.label)] += 1
    return counts


def class_scores(
    counts: Counter[tuple[str, str]], classes: Sequence[str]
) -> tuple[dict[str, ClassScore], ClassScore]:
    """The score of each of ``classes``, and one for any box whatever its label.

    For a class ``c``, ``reference`` counts the pairs ``(c, anything)``,
    ``detected`` the pairs ``(anything, c)`` and ``matched`` the pairs ``(c, c)``:
    a wrong label is a miss for the class that was annotated and a false box for
    the class that was given. The second value counts every pair with a box on both
    sides as matched, labels ignored: how well the boxes are found, apart from how
    well they are named. A label that is not in ``classes`` is in that line only.

    Raises:
        ValueError: ``classes`` holds ``NO_BOX``, which would be read as a missing box.
    """
    if NO_BOX in classes:
        raise ValueError(
            f"{NO_BOX!r} marks a missing box and cannot be the name of a class"
        )
    scores = {
        name: _score(
            reference=sum(n for (ref, _), n in counts.items() if ref == name),
            detected=sum(n for (_, det), n in counts.items() if det == name),
            matched=counts[(name, name)],
        )
        for name in classes
    }
    anything = _score(
        reference=sum(n for (ref, _), n in counts.items() if ref != NO_BOX),
        detected=sum(n for (_, det), n in counts.items() if det != NO_BOX),
        matched=sum(
            n for (ref, det), n in counts.items() if ref != NO_BOX and det != NO_BOX
        ),
    )
    return scores, anything


def _score(*, reference: int, detected: int, matched: int) -> ClassScore:
    """The ratios, ``None`` where a denominator is 0 (never 0 and never ``nan``)."""
    precision = matched / detected if detected > 0 else None
    recall = matched / reference if reference > 0 else None
    return ClassScore(reference, detected, matched, precision, recall)
