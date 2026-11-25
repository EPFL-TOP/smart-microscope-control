"""What the Stand screen says of a calibration's numbers: the text, the bound, the rule and the time (m5 §23, #627).

No Panel and no device here. The text is ``calibration_numbers_text`` on a ``CalibrationNote``; the
rule (which calibration the page gives the numbers of) is ``stand_calibration`` on the data it is
given. Every expected text is the design's, written out whole and not built by the code under test
(FM-45); a number is the one the demo measured (m5 §23.1) and not one the code computes (FM-203).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from smc.ui import controller
from smc.ui.guide import (
    CALIBRATION_LINE_CHARS,
    CalibrationNote,
    CalibrationNumbers,
    calibration_numbers_text,
    local_time,
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


@pytest.mark.parametrize(
    ("length", "cut"),
    [
        pytest.param(CALIBRATION_LINE_CHARS - 1, False, id="one under the bound"),
        pytest.param(CALIBRATION_LINE_CHARS, False, id="at the bound"),
        pytest.param(CALIBRATION_LINE_CHARS + 1, True, id="one over the bound"),
        pytest.param(CALIBRATION_LINE_CHARS + 2, True, id="two over the bound"),
    ],
)
def test_a_line_of_the_bound_is_kept_whole_and_one_character_more_is_cut(
    length: int, cut: bool
) -> None:
    # The edge itself (FM-43): `assumed: ` is 9 characters, the note is the rest.
    prefix = "assumed: "
    note = "n" * (length - len(prefix))

    line = calibration_numbers_text(_note(notes=(note,))).split("\n")[-1]

    if cut:
        assert line == (prefix + note)[:CALIBRATION_LINE_CHARS] + "…"
    else:
        assert line == prefix + note
        assert len(line) == length


def test_the_line_bound_is_the_cards_sentence_bound() -> None:
    assert CALIBRATION_LINE_CHARS == controller._CARD_TEXT_CHARS == 600


# --- which calibration the pane gives the numbers of (D1) ----------------------------------

TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
#: The fake turret's labels as the page reads them, empty positions included: the 40X is position
#: 1, the 10X position 2, the 20X position 4 (the index plus one, written here by hand).
LABELS = (FORTY, TEN, "Objective-2", TWENTY, "Objective-4", "Objective-5")
WHEN = "2026-11-25T10:20:30+00:00"
#: The run document's calibration, at the 10X: a note of its own, distinct from every document's.
RUN_NUMBERS = CalibrationNumbers(0.6500, 1.0, False, 0.5, 48.0)
RUN = CalibrationNote(
    TEN,
    "2026-11-25 09:00:00",
    "2026-11-25 10:00",
    RUN_NUMBERS,
    ("the run document's own line",),
    "run/run.json",
)


def _read(
    objective: str | None = FORTY,
    *,
    path: str = "run/calibrations/position-1.json",
    measured: str = WHEN,
    problem: str | None = None,
    numbers: CalibrationNumbers | None = DEMO,
    notes: tuple[str, ...] = ("a line of the document",),
) -> controller.CalibrationRead:
    return controller.CalibrationRead(
        path,
        objective,
        measured,
        problem,
        notes,
        image_shape_px=(384, 512),
        numbers=numbers,
    )


def _stand(**changes: object) -> CalibrationNote | None:
    """``stand_calibration`` for the 40X after a registration under the 10X, with the 40X's document held."""
    arguments: dict[str, object] = {
        "objective": FORTY,
        "plate_objective": TEN,
        "run_calibration": RUN,
        "labels": LABELS,
        "documents": (controller.HeldCalibration(1, _read()),),
    }
    arguments.update(changes)
    return controller.stand_calibration(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("objective", [None, "", "   "])
def test_the_stand_calibration_with_the_objective_not_known_is_the_run_documents(
    objective: str | None,
) -> None:
    # A closed stand, a turret that does not answer, an empty position: what the known
    # block's `Camera:` line speaks of, whichever objective the document recorded.
    assert _stand(objective=objective) is RUN
    other = CalibrationNote(FORTY, "2026-11-25 09:00:00", numbers=RUN_NUMBERS)
    assert _stand(objective=objective, run_calibration=other) is other
    assert _stand(objective=objective, run_calibration=None) is None


@pytest.mark.parametrize(
    ("plate_objective", "objective", "recorded", "shown"),
    [
        pytest.param(TEN, TEN, TEN, True, id="the plate's own objective"),
        pytest.param(None, TEN, TEN, True, id="no offer, the objective it was made at"),
        pytest.param(TEN, TEN, None, True, id="the plate's own, none recorded"),
        pytest.param(None, FORTY, None, True, id="no offer, none recorded"),
        pytest.param(None, FORTY, TEN, False, id="no offer, another objective's"),
        pytest.param(TEN, TEN, FORTY, False, id="the plate's own, another's"),
    ],
)
def test_at_the_plates_objective_or_with_no_offer_the_run_documents_calibration_is_shown_only_for_that_objective(
    plate_objective: str | None, objective: str, recorded: str | None, shown: bool
) -> None:
    run = CalibrationNote(recorded, "2026-11-25 09:00:00", "", RUN_NUMBERS)
    # A held document at the objective's own position must never be taken in these cases.
    documents = (controller.HeldCalibration(1, _read()),)

    found = _stand(
        plate_objective=plate_objective,
        objective=objective,
        run_calibration=run,
        documents=documents,
    )

    assert found == (run if shown else None)


def test_at_the_plates_own_objective_a_document_held_at_its_position_is_not_the_run_documents() -> (
    None
):
    # The click at the plate's own objective writes into the run document (D19), so a file left at
    # that objective's position by an earlier session is not the calibration in use: the pane says
    # the run document's, with its own numbers and path (FM-216; the case D1 item 2 is for).
    stale = _read(
        TEN,
        path="run/calibrations/position-2.json",
        numbers=DEMO,
        notes=("a stale file",),
    )

    found = _stand(
        objective=TEN,
        plate_objective=TEN,
        # The 10X is the second label of LABELS: position 2.
        documents=(controller.HeldCalibration(2, stale),),
    )

    assert found is RUN
    assert found is not None
    assert found.where == "run/run.json"
    assert found.numbers == RUN_NUMBERS


def test_at_another_objective_the_positions_document_is_shown() -> None:
    documents = (
        controller.HeldCalibration(
            1, _read(FORTY, path="c/position-1.json", numbers=DEMO, notes=("forty",))
        ),
        controller.HeldCalibration(
            4,
            _read(
                TWENTY,
                path="c/position-4.json",
                numbers=RUN_NUMBERS,
                notes=("twenty", "twenty again"),
            ),
        ),
    )

    at_twenty = _stand(objective=TWENTY, documents=documents)
    at_forty = _stand(objective=FORTY, documents=documents)

    assert at_twenty == CalibrationNote(
        TWENTY,
        "2026-11-25 10:20:30",
        local_time(datetime(2026, 11, 25, 10, 20, 30, tzinfo=timezone.utc)),
        RUN_NUMBERS,
        ("twenty", "twenty again"),
        "c/position-4.json",
    )
    assert at_forty == CalibrationNote(
        FORTY,
        "2026-11-25 10:20:30",
        local_time(datetime(2026, 11, 25, 10, 20, 30, tzinfo=timezone.utc)),
        DEMO,
        ("forty",),
        "c/position-1.json",
    )
    # Not the run document's, which is the 10X's.
    assert at_twenty != RUN
    assert text_of(at_forty).splitlines()[2] == "Written to c/position-1.json."


def text_of(note: CalibrationNote | None) -> str:
    assert note is not None
    return calibration_numbers_text(note)


@pytest.mark.parametrize(
    "held",
    [
        pytest.param(
            controller.HeldCalibration(1, _read(TEN)),
            id="the document holds another objective's calibration",
        ),
        pytest.param(
            controller.HeldCalibration(
                1, _read(None, problem="position 1: it is not JSON", numbers=None)
            ),
            id="a document with a problem",
        ),
        pytest.param(
            controller.HeldCalibration(1, _read(FORTY, numbers=None)),
            id="a read with no numbers",
        ),
        pytest.param(
            controller.HeldCalibration(2, _read(FORTY)),
            id="the document of another position",
        ),
    ],
)
def test_a_document_that_is_not_its_positions_has_a_problem_or_no_numbers_gives_nothing(
    held: controller.HeldCalibration,
) -> None:
    assert _stand(documents=(held,)) is None


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"labels": ()}, id="the turret's labels were not read"),
        pytest.param({"objective": "Objective-9"}, id="an objective not on the turret"),
        pytest.param({"documents": ()}, id="no document held"),
        pytest.param(
            {"labels": (FORTY, FORTY, TEN)},
            id="two positions carry the label: which is in place is not said",
        ),
    ],
)
def test_a_position_the_page_cannot_name_or_a_missing_document_gives_nothing(
    changes: dict[str, object],
) -> None:
    assert _stand(**changes) is None


def test_the_rule_never_shows_the_run_documents_numbers_as_another_objectives() -> None:
    # FM-216, the headline: an offer under the 10X, the 40X in place and nothing held for it.
    assert _stand(documents=()) is None
    assert _stand(documents=(), run_calibration=RUN) is None


# --- the time of a document (D4) ----------------------------------------------------------


def test_the_time_of_a_document_is_the_tools_own_iso_text_and_an_unknown_time_is_said() -> (
    None
):
    instant = datetime(2026, 11, 25, 10, 20, 30, tzinfo=timezone.utc)
    # The form `calibration_from` returns, built here the way it builds it.
    iso = instant.isoformat(timespec="seconds")

    assert controller._measured_times(iso) == (
        "2026-11-25 10:20:30",
        local_time(instant),
    )
    assert controller._measured_times("at an unknown time") == (
        "????-??-?? ??:??:??",
        "",
    )
    assert controller._measured_times("yesterday") == ("????-??-?? ??:??:??", "")
    assert controller._measured_times("") == ("????-??-?? ??:??:??", "")


@pytest.fixture
def pc_zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The PC set to UTC+14, where no clock reading of a UTC time can be mistaken for the PC's.

    On a runner in UTC a naive time read as the PC's and the same time read as UTC are one text,
    so the rule "a time with no zone is UTC" is only seen from another zone (FM-43). ``tzset`` is
    the platform's: Windows reads the zone from the system and the test is skipped there.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("this platform sets no zone from TZ")
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    time.tzset()
    try:
        yield "Pacific/Kiritimati"
    finally:
        monkeypatch.undo()
        time.tzset()


def test_a_time_in_another_zone_or_with_none_is_put_in_utc() -> None:
    plus_two = "2026-11-25T12:20:30+02:00"
    naive = "2026-11-25T10:20:30"  # a run document's times are UTC: never the PC's zone
    instant = datetime(2026, 11, 25, 10, 20, 30, tzinfo=timezone.utc)

    assert controller._measured_times(plus_two) == (
        "2026-11-25 10:20:30",
        local_time(instant),
    )
    assert controller._measured_times(naive) == (
        "2026-11-25 10:20:30",
        local_time(instant),
    )


def test_a_time_with_no_zone_is_utc_whatever_zone_the_pc_is_in(pc_zone: str) -> None:
    naive = "2026-11-25T10:20:30"
    instant = datetime(2026, 11, 25, 10, 20, 30, tzinfo=timezone.utc)

    utc, local = controller._measured_times(naive)

    # The PC is 14 hours ahead of UTC: the local form is the next morning, the UTC form is as written.
    assert utc == "2026-11-25 10:20:30"
    assert local == local_time(instant) == "2026-11-26 00:20"


def test_a_time_utc_cannot_hold_is_the_unknown_time_and_raises_nothing() -> None:
    # Year 1 with an offset: a model accepts it, UTC cannot hold it (FM-98, FM-125).
    utc, _local = controller._measured_times("0001-01-01T00:00:00+05:00")

    assert utc == "????-??-?? ??:??:??"
