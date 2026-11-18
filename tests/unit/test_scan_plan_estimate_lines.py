"""The two ``assumed`` lines a scan plan's estimate carries, and the bound the page shows them through (#549, m3 §7.4 D5).

The words are fixed by the design, character for character; the test reads them
whole (FM-45) and measures the longest of each against the page's bound
(FM-205), which lives in ``smc.ui.guide`` and which ``smc.plugins`` does not import.
"""

from __future__ import annotations

from smc.plugins.scan_plan.algorithm import estimate_assumed
from smc.ui.guide import ASSUMED_LINE_CHARS

LOCKS = (
    "estimate: the autofocus holds its lock between fields and locks again once "
    "in each well ({n}); one that refocuses at each field would take longer"
)
LEFT_OUT_WITH_LOCKS = (
    "estimate: the timed moves, locks and snaps; it leaves out each frame's "
    "write, a lock lost in a well, a channel change, the cells, the report and "
    "the export"
)
LEFT_OUT_WITHOUT_LOCKS = (
    "estimate: the timed moves and snaps; it leaves out each frame's write, a "
    "channel change, the cells, the report and the export"
)


def test_the_lock_line_names_the_count_and_says_what_it_assumes() -> None:
    assert estimate_assumed(2)[0] == LOCKS.format(n="2 locks")
    assert estimate_assumed(1)[0] == LOCKS.format(n="1 lock")
    assert estimate_assumed(72)[0] == LOCKS.format(n="72 locks")
    assert estimate_assumed(1)[0].endswith(
        "(1 lock); one that refocuses at each field would take longer"
    )


def test_the_left_out_line_is_one_of_two_and_each_fits_the_page_whole() -> None:
    # With locks: the lock line, then the left-out line that names locks.
    assert estimate_assumed(2) == (LOCKS.format(n="2 locks"), LEFT_OUT_WITH_LOCKS)
    # With none: one line, which names none.
    assert estimate_assumed(0) == (LEFT_OUT_WITHOUT_LOCKS,)
    # The longest forms, at the largest count the design measured (FM-205): the
    # page cuts a line at ASSUMED_LINE_CHARS, and these were measured at 152 and 155.
    longest = estimate_assumed(9999)
    assert [len(line) for line in longest] == [152, 155]
    assert all(len(line) <= ASSUMED_LINE_CHARS for line in longest)
    assert len(estimate_assumed(0)[0]) <= ASSUMED_LINE_CHARS
