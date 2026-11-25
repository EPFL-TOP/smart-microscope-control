"""What the Stand screen says of a calibration's numbers: the text, the bound, the rule and the time (m5 §23, #627).

No Panel and no device here. The text is ``calibration_numbers_text`` on a ``CalibrationNote``; the
rule (which calibration the page gives the numbers of) is ``stand_calibration`` on the data it is
given. Every expected text is the design's, written out whole and not built by the code under test
(FM-45); a number is the one the demo measured (m5 §23.1) and not one the code computes (FM-203).
"""

from __future__ import annotations

import pytest

from smc.ui import controller
from smc.ui.guide import (
    CALIBRATION_LINE_CHARS,
    CalibrationNote,
    CalibrationNumbers,
    calibration_numbers_text,
)

FORTY = "Nikon 40X Plan Fluor ELWD"
#: What ``demo-sample`` turned as §18.6.1 measured (m5 §23.1), as the document holds them.
DEMO = CalibrationNumbers(
    pixel_size_um=0.249966,
    rotation_deg=29.9891,
    mirrored=True,
    residual_um=0.0437417,
    step_um=32.0,
)
DEMO_FIRST = (
    "Camera calibration at Nikon 40X Plan Fluor ELWD, measured 2026-11-25 12:05."
)
DEMO_SECOND = "Pixel size 0.2500 µm, rotation 30.0°, mirrored, residual 0.04 µm, probe step 32 µm."
DEMO_WHERE = "run/calibrations/position-1.json"


def _note(**changes: object) -> CalibrationNote:
    """The demo's note: the 40X, measured at 12:05 on the PC, the numbers and the file."""
    arguments: dict[str, object] = {
        "objective": FORTY,
        "when_utc": "2026-11-25 11:05:00",
        "when_local": "2026-11-25 12:05",
        "numbers": DEMO,
        "notes": (),
        "where": DEMO_WHERE,
    }
    arguments.update(changes)
    return CalibrationNote(**arguments)  # type: ignore[arg-type]


# --- the text (D3) ---------------------------------------------------------------------


def test_the_numbers_text_says_objective_time_numbers_and_file_whole() -> None:
    assert calibration_numbers_text(_note()) == "\n".join(
        [DEMO_FIRST, DEMO_SECOND, "Written to run/calibrations/position-1.json."]
    )


def test_the_calibrations_own_assumed_lines_follow_the_three_lines() -> None:
    notes = (
        "the pixel size 0.25 µm is the profile's, not measured",
        "residual 1.8 µm is more than 5 % of the 32 µm step",
    )

    text = calibration_numbers_text(_note(notes=notes))

    assert text.split("\n") == [
        DEMO_FIRST,
        DEMO_SECOND,
        "Written to run/calibrations/position-1.json.",
        "assumed: the pixel size 0.25 µm is the profile's, not measured",
        "assumed: residual 1.8 µm is more than 5 % of the 32 µm step",
    ]


def test_a_calibration_with_no_objective_and_no_file_says_so_and_leaves_the_third_line_out() -> (
    None
):
    text = calibration_numbers_text(_note(objective=None, where=""))

    assert text.split("\n") == [
        "Camera calibration (objective not recorded), measured 2026-11-25 12:05.",
        DEMO_SECOND,
    ]


def test_a_time_the_pc_has_no_local_form_for_is_said_in_utc() -> None:
    text = calibration_numbers_text(_note(when_local=""))

    assert text.split("\n")[0] == (
        "Camera calibration at Nikon 40X Plan Fluor ELWD, measured "
        "2026-11-25 11:05:00 UTC."
    )


@pytest.mark.parametrize(
    ("rotation", "said"),
    [
        pytest.param(-0.0, "rotation 0.0°", id="negative zero"),
        pytest.param(-0.04, "rotation 0.0°", id="rounds to negative zero"),
        pytest.param(0.0, "rotation 0.0°", id="zero"),
        pytest.param(-12.34, "rotation -12.3°", id="a negative angle keeps its sign"),
        pytest.param(180.0, "rotation 180.0°", id="the bound"),
        pytest.param(-180.0, "rotation -180.0°", id="the other bound"),
    ],
)
def test_a_negative_zero_rotation_reads_zero(rotation: float, said: str) -> None:
    numbers = CalibrationNumbers(0.25, rotation, True, 0.04, 32.0)

    second = calibration_numbers_text(_note(numbers=numbers)).split("\n")[1]

    assert (
        second
        == f"Pixel size 0.2500 µm, {said}, mirrored, residual 0.04 µm, probe step 32 µm."
    )


def test_the_mirror_is_said_both_ways() -> None:
    numbers = CalibrationNumbers(0.25, 30.0, False, 0.04, 32.0)

    second = calibration_numbers_text(_note(numbers=numbers)).split("\n")[1]

    assert second == (
        "Pixel size 0.2500 µm, rotation 30.0°, not mirrored, residual 0.04 µm, "
        "probe step 32 µm."
    )


def test_a_probe_step_is_said_as_the_tool_keeps_it() -> None:
    numbers = CalibrationNumbers(0.25, 30.0, True, 0.04, 7.5)

    second = calibration_numbers_text(_note(numbers=numbers)).split("\n")[1]

    assert second.endswith("probe step 7.5 µm.")


def test_a_note_with_no_numbers_gives_no_text() -> None:
    assert calibration_numbers_text(_note(numbers=None, notes=("x",))) == ""


# --- plain, one line each, bounded (D5) ---------------------------------------------------


def test_a_hand_edited_value_cannot_make_a_line_longer_than_the_bound() -> None:
    huge = CalibrationNumbers(1e300, 29.9891, True, 1e300, 32.0)

    lines = calibration_numbers_text(_note(numbers=huge)).split("\n")

    assert len(lines) == 3
    assert all(len(line) <= CALIBRATION_LINE_CHARS + 1 for line in lines)
    assert lines[1].endswith("…")
    assert lines[1].startswith("Pixel size 1000000")
    assert lines[0] == DEMO_FIRST
    assert lines[2] == "Written to run/calibrations/position-1.json."


def test_a_label_a_path_and_a_note_are_cut_each_at_the_bound() -> None:
    long = "x" * 5000

    text = calibration_numbers_text(
        _note(objective=long, where=long, notes=(long, long))
    )

    lines = text.split("\n")
    assert len(lines) == 5
    assert [len(line) for line in lines] == [
        CALIBRATION_LINE_CHARS + 1,
        len(DEMO_SECOND),
        CALIBRATION_LINE_CHARS + 1,
        CALIBRATION_LINE_CHARS + 1,
        CALIBRATION_LINE_CHARS + 1,
    ]
    assert all(line.endswith("…") for line in lines if line != DEMO_SECOND)


def test_a_note_and_a_label_with_line_breaks_stay_on_their_own_line() -> None:
    text = calibration_numbers_text(
        _note(
            objective="Nikon\n40X",
            where="a\r\nb",
            notes=("two\nlines\tand\t\ttabs",),
        )
    )

    assert text.split("\n") == [
        "Camera calibration at Nikon 40X, measured 2026-11-25 12:05.",
        DEMO_SECOND,
        "Written to a  b.",
        "assumed: two lines and tabs",
    ]


def test_the_fixed_text_is_never_cut_at_the_widest_value_the_model_allows() -> None:
    """FM-205: the bound is for what nobody wrote; what the tool writes reads whole."""
    widest = CalibrationNumbers(99999.9999, -180.0, False, 99999.99, 99999.0)

    second = calibration_numbers_text(_note(numbers=widest)).split("\n")[1]

    assert second == (
        "Pixel size 99999.9999 µm, rotation -180.0°, not mirrored, residual "
        "99999.99 µm, probe step 99999 µm."
    )
    assert len(second) < CALIBRATION_LINE_CHARS


def test_the_line_bound_is_the_cards_sentence_bound() -> None:
    assert CALIBRATION_LINE_CHARS == controller._CARD_TEXT_CHARS == 600
