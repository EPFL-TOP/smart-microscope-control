"""The two texts a stop's score is written with, on the map's hover and in the report (m5 §21, D9).

One rule for a score, shared by the page, the report and the drawing of the data
module, so that a number read in one place is the number read in another.
"""

from __future__ import annotations

import pytest

from smc.data.draw import label_of
from smc.plugins.slots import measures_text, score_text


def test_score_text_is_three_significant_digits_and_a_score_of_zero_is_a_score() -> (
    None
):
    assert score_text(0.0) == "0"
    assert score_text(0.823456) == "0.823"
    assert score_text(1234.5678) == "1.23e+03"
    assert score_text(-0.5) == "-0.5"


def test_measures_text_is_sorted_by_key_and_stops_at_the_limit() -> None:
    assert measures_text({"b_um": 2.0, "a": 1.0}) == "a 1, b_um 2"
    eight = {f"k{i}": float(i) for i in range(8)}
    assert measures_text(eight) == "k0 0, k1 1, k2 2, k3 3, k4 4, k5 5, +2 more"
    assert measures_text({}) == ""


def test_measures_text_exactly_at_the_limit_has_no_more_and_the_limit_is_a_parameter() -> (
    None
):
    six = {f"k{i}": 0.5 for i in range(6)}
    assert measures_text(six) == "k0 0.5, k1 0.5, k2 0.5, k3 0.5, k4 0.5, k5 0.5"
    assert measures_text(six, limit=2) == "k0 0.5, k1 0.5, +4 more"


@pytest.mark.parametrize("score", [0.0, 0.823456, 1234.5678, 0.00012345])
def test_score_text_is_the_drawings_rule(score: float) -> None:
    assert label_of(3, score) == f"3 ({score_text(score)})"
