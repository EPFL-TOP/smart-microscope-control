"""The arithmetic of the plate check: which two wells, and how the plate moved (m3 §27.3).

Pure: ``math`` only, no device and no file. At Ti2-E session 5 (#550, F15) a
plate registered by ``plate-walls`` was lifted and put back; nothing noticed,
and the 40x scan made then placed its frames about 50 µm off in their wells.
The check measures two wells again and compares their centres with the
registered ones. Which two: the pair farthest apart, since one well sees a
slide and not a turn about itself, and the farthest pair sees a turn at its
longest lever (§27.2 D1). The turn and the slide are said to the operator and
never judged: the verdict is how far a well moved (D3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "DEFAULT_MAX_SHIFT_UM",
    "DEFAULT_SEARCH_UM",
    "farthest_pair",
    "rigid_motion",
]

#: How far a wall may lie from where the registered centre puts it, at most.
DEFAULT_SEARCH_UM = 1000.0
#: The largest distance a well may have moved (§27.2 D3): session 5's move read
#: 31.9 and 33.5 µm at its farthest pair, the scatter of one objective's offset
#: across wells reached 10.9 µm (F17). Chosen; revisited at the stand (§27.9).
DEFAULT_MAX_SHIFT_UM = 25.0


def farthest_pair(
    centres: Sequence[tuple[str, tuple[float, float]]],
) -> tuple[str, str]:
    """The two names whose centres are farthest apart, in the order given.

    On a tie the first pair met in the order ``(i, j)``, ``i < j``, is kept: a
    strict ``>`` never replaces it, so the registration's order decides and the
    choice is the same on every run.

    Raises:
        ValueError: Fewer than two centres.
    """
    if len(centres) < 2:
        raise ValueError(f"two wells are needed to choose a pair, got {len(centres)}")
    best: tuple[str, str] = (centres[0][0], centres[1][0])
    longest = -1.0
    for i, (first, a) in enumerate(centres):
        for second, b in centres[i + 1 :]:
            distance = math.dist(a, b)
            if distance > longest:
                longest = distance
                best = (first, second)
    return best


def rigid_motion(
    expected: Sequence[tuple[float, float]],
    measured: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float]]:
    """The least-squares turn (degrees) and slide that carry ``expected`` onto ``measured``.

    The 2-D form of Kabsch: with both sets centred on their means, the turn is
    ``atan2(Σ (e_x m_y - e_y m_x), Σ (e_x m_x + e_y m_y))`` and the slide is the
    mean of ``measured`` minus the mean of ``expected``. For two wells the turn
    is the angle between the two vectors that join them. The slide is of the
    mean, not of the origin: a turn about the plate's centre moves the stage's
    origin by nothing and a well 58 mm away by tens of µm.

    Raises:
        ValueError: Fewer than two points, lengths that differ, or a value that
            is not finite.
    """
    if len(expected) != len(measured):
        raise ValueError(
            f"as many measured centres as expected ones are needed, got "
            f"{len(measured)} and {len(expected)}"
        )
    if len(expected) < 2:
        raise ValueError(f"two points are needed to read a turn, got {len(expected)}")
    for x, y in (*expected, *measured):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"a centre must be finite, got ({x!r}, {y!r})")
    n = len(expected)
    ex = sum(x for x, _ in expected) / n
    ey = sum(y for _, y in expected) / n
    mx = sum(x for x, _ in measured) / n
    my = sum(y for _, y in measured) / n
    cross = 0.0
    dot = 0.0
    for (ax, ay), (bx, by) in zip(expected, measured, strict=True):
        ax, ay, bx, by = ax - ex, ay - ey, bx - mx, by - my
        cross += ax * by - ay * bx
        dot += ax * bx + ay * by
    return math.degrees(math.atan2(cross, dot)), (mx - ex, my - ey)
