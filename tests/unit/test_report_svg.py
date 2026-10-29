"""The drawing primitives of the report (m3 §13, #195): numbers, escaping, colours, the subsample."""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.plugins.report.svg import (
    LABEL_COLOURS,
    density_text,
    escape,
    label_colour,
    num,
    sample_indices,
    shade,
    text_colour_on,
)


def _contrast(foreground: str, background: str) -> float:
    """The WCAG 2 contrast ratio, written here from the definition and not taken from the code."""

    def luminance(colour: str) -> float:
        parts = [int(colour[k : k + 2], 16) / 255.0 for k in (1, 3, 5)]
        red, green, blue = (
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in parts
        )
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    lighter, darker = sorted(
        (luminance(foreground), luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_num_rounds_to_a_tenth_and_never_writes_minus_zero() -> None:
    kw = {"what": "a frame's position", "stage": "scan"}
    assert num(12.34, **kw) == "12.3"
    assert num(12.36, **kw) == "12.4"
    assert num(-45.6, **kw) == "-45.6"
    # A trailing .0 is dropped: it is 16 bytes a dot on a page of 300 000.
    assert num(100.0, **kw) == "100"
    assert num(0.0, **kw) == "0"
    # f"{-0.04:.1f}" is "-0.0": a sign that is not there.
    assert num(-0.04, **kw) == "0"
    assert num(-0.0, **kw) == "0"
    assert num(0.04, **kw) == "0"
    assert num(-0.06, **kw) == "-0.1"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "a cell's position is not a finite number; run cells again"),
        (float("inf"), "a cell's position is not a finite number; run cells again"),
        (float("-inf"), "a cell's position is not a finite number; run cells again"),
        (1e300, "a cell's position is beyond 1e12 µm; run cells again"),
        (-1e13, "a cell's position is beyond 1e12 µm; run cells again"),
    ],
)
def test_num_refuses_what_is_not_finite_or_beyond_1e12_naming_the_stage(
    value: float, message: str
) -> None:
    # The whole message: anchored at both ends.
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        num(value, what="a cell's position", stage="cells")


def test_num_accepts_the_bound_itself() -> None:
    assert num(1e12, what="x", stage="scan") == "1000000000000"


def test_escape_quotes_both_kinds_of_quote_and_markup() -> None:
    assert (
        escape("<b>\"x\" & 'y'</b>")
        == "&lt;b&gt;&quot;x&quot; &amp; &#x27;y&#x27;&lt;/b&gt;"
    )
    # Anything is text: a number and None are written, not refused.
    assert escape(12) == "12"
    assert escape(None) == "None"


def test_sample_indices_are_exactly_cap_distinct_and_increasing() -> None:
    assert sample_indices(10, 2000) == list(range(10))
    assert sample_indices(2000, 2000) == list(range(2000))
    for n in (5000, 5001, 20_000):
        chosen = sample_indices(n, 2000)
        assert len(chosen) == 2000
        assert chosen == sorted(set(chosen))
        assert chosen[0] >= 0
        assert chosen[-1] < n
    # The first is always the first cell, and an even thinning of 5 by 2 is 0 and 2.
    assert sample_indices(5, 2) == [0, 2]
    assert sample_indices(7, 3) == [0, 2, 4]


def test_sample_indices_of_a_cap_of_zero_are_none_and_a_negative_n_is_refused() -> None:
    assert sample_indices(10, 0) == []
    assert sample_indices(0, 0) == []
    assert sample_indices(0, 5) == []
    with pytest.raises(ValueError, match="cannot draw a sample of -1 items"):
        sample_indices(-1, 5)


@given(n=st.integers(0, 50_000), cap=st.integers(1, 5_000))
def test_sample_indices_never_exceed_the_cap_and_are_distinct_increasing(
    n: int, cap: int
) -> None:
    chosen = sample_indices(n, cap)
    assert len(chosen) == min(n, cap)
    assert chosen == sorted(set(chosen))
    assert all(0 <= k < n for k in chosen)


def test_label_colour_cycles_through_the_seven_and_agrees_with_the_set() -> None:
    assert label_colour(0) == "#D55E00"
    assert label_colour(6) == "#7A7A7A"
    assert label_colour(7) == label_colour(0)
    assert [label_colour(k) for k in range(7)] == list(LABEL_COLOURS)


def test_text_colour_on_every_step_of_every_label_reads_at_4_5_to_1() -> None:
    for colour in LABEL_COLOURS:
        for step in range(21):
            fill = shade(colour, step / 20)
            assert _contrast(text_colour_on(fill), fill) >= 4.5, (colour, step, fill)


def test_text_colour_on_picks_black_on_light_and_white_on_dark() -> None:
    assert text_colour_on("#ffffff") == "#000000"
    assert text_colour_on("#000000") == "#FFFFFF"


def test_shade_of_zero_is_not_white_and_of_one_is_the_colour() -> None:
    # mix 0.10: 255 + (c - 255) * 0.10 per channel, rounded: 250.8, 238.9, 229.5.
    assert shade("#D55E00", 0.0) == "#fbefe6"
    assert shade("#D55E00", 1.0) == "#d55e00"
    assert shade("#0072B2", 1.0) == "#0072b2"
    assert shade("#D55E00", 0.0) != "#ffffff"
    # Halfway: mix 0.55, 255 - 140.25 = 114.75.
    assert shade("#000000", 0.5) == "#737373"


def test_shade_clamps_t_and_reads_a_value_that_is_not_finite_as_zero() -> None:
    assert shade("#D55E00", -3.0) == shade("#D55E00", 0.0)
    assert shade("#D55E00", 7.0) == shade("#D55E00", 1.0)
    assert shade("#D55E00", float("nan")) == shade("#D55E00", 0.0)
    assert shade("#D55E00", float("inf")) == shade("#D55E00", 0.0)


def test_density_text_has_no_exponent() -> None:
    assert density_text(1234.5) == "1234"
    assert density_text(100.0) == "100"
    assert density_text(99.4) == "99.4"
    assert density_text(10.0) == "10.0"
    assert density_text(9.996) == "10.00"
    assert density_text(0.12) == "0.12"
    assert density_text(0.004) == "0.00"
    assert density_text(1e9) == "1000000000"
    for value in (1234.5, 99.95, 1e9, 0.004, 1e-9, 1e15):
        assert "e" not in density_text(value)


def test_density_text_of_a_measured_zero_is_a_bare_zero() -> None:
    assert density_text(0.0) == "0"
