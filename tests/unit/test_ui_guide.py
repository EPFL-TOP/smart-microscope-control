"""The next step's pure pieces: which slot is next, and the command that makes its call (m5 §8.3, #293).

No Panel and no stand. The command is read back with ``shlex.split``, the way a
shell reads it, never compared with the string the code builds (FM-45).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.ui.controller import CheckItem
from smc.ui.guide import (
    CAPTURE_WELLS,
    SELECT_WELLS,
    NextStep,
    command_line,
    next_slot,
)


def _item(
    slot: str,
    state: str = "not run",
    *,
    optional: bool = False,
    reason: str | None = None,
) -> CheckItem:
    return CheckItem(slot, optional, state, reason)  # type: ignore[arg-type]


def test_next_slot_skips_optional_slots_and_done_ones() -> None:
    checklist = (
        _item("camera-calibration", optional=True),
        _item("plate", "ok"),
        _item("timings", optional=True),
        _item("scan-plan"),
        _item("scan"),
    )
    found = next_slot(checklist)
    assert found is not None
    assert found.slot == "scan-plan"


def test_next_slot_stops_at_an_unusable_one_with_its_reason() -> None:
    reason = "the last run of plate-registration ended failed; run it again"
    checklist = (
        _item("plate", "unusable", reason=reason),
        _item("scan-plan"),
    )
    found = next_slot(checklist)
    assert found is not None
    assert (found.slot, found.state, found.reason) == ("plate", "unusable", reason)


def test_next_slot_is_none_when_every_required_slot_is_ok() -> None:
    checklist = (
        _item("camera-calibration", optional=True),
        _item("plate", "ok"),
        _item("timings", "unusable", optional=True, reason="stale"),
        _item("scan", "ok"),
        # A third party's slot is optional: the workflow does not wait for it.
        _item("their-slot", optional=True),
    )
    assert next_slot(checklist) is None
    assert next_slot(()) is None


def test_next_slot_takes_the_first_of_several_missing_in_the_checklists_order() -> None:
    checklist = (_item("plate"), _item("scan-plan"), _item("scan"))
    found = next_slot(checklist)
    assert found is not None
    assert found.slot == "plate"


def test_the_two_waiting_texts_are_the_designs() -> None:
    assert SELECT_WELLS == (
        "select the wells to scan: click them on the map, or type a range in Wells "
        "(A1:D6, C*, all)"
    )
    assert CAPTURE_WELLS == (
        "centre a well under the objective (Live and the Drive pad), click it on the "
        "map and press Capture centre; a second well, far from the first, fits the "
        "rotation too"
    )


def test_a_next_step_is_frozen_and_holds_what_it_was_given() -> None:
    step = NextStep(
        plugin="scan",
        params=("a=1",),
        why=("the run document has no scan yet",),
        command="smc run scan",
        blocked=None,
    )
    assert (step.plugin, step.params, step.blocked) == ("scan", ("a=1",), None)
    # The page keeps the step it drew and runs it at the click: it must not change.
    with pytest.raises(AttributeError):
        step.plugin = "other"  # type: ignore[misc]


_REFS = json.dumps(
    [
        {"well": "A1", "x_um": -49500.0, "y_um": 31500.0},
        {"well": "H1", "x_um": -49500.0, "y_um": -31500.0},
    ]
)


def test_command_line_round_trips_json_through_shlex() -> None:
    pairs = ["plate=96-well", f"refs={_REFS}", "walls=[]", 'note=it\'s "quoted"']
    line = command_line(
        "plate-registration",
        pairs,
        profile="demo-sample",
        run_doc=Path("/runs/exp one/run.json"),
        dry_run=False,
    )
    words = shlex.split(line)
    assert words[:3] == ["smc", "run", "plate-registration"]
    taken = [words[i + 1] for i, word in enumerate(words) if word == "--param"]
    assert taken == pairs
    # The JSON is one word and parses back to what was dumped.
    assert json.loads(taken[1].removeprefix("refs=")) == json.loads(_REFS)


def test_command_line_names_the_profile_the_dry_run_and_the_run_doc() -> None:
    run_doc = Path("/runs/exp one/run.json")
    wet = shlex.split(
        command_line(
            "scan",
            ["a=1"],
            profile="/p/demo sample.toml",
            run_doc=run_doc,
            dry_run=False,
        )
    )
    assert wet == [
        "smc",
        "run",
        "scan",
        "-p",
        "/p/demo sample.toml",
        "--run-doc",
        str(run_doc),
        "--param",
        "a=1",
    ]
    dry = shlex.split(
        command_line("scan", [], profile="demo", run_doc=run_doc, dry_run=True)
    )
    assert dry == [
        "smc",
        "run",
        "scan",
        "-p",
        "demo",
        "--run-doc",
        str(run_doc),
        "--dry-run",
    ]


@given(
    st.lists(
        st.tuples(
            st.from_regex(r"[a-z_]{1,8}", fullmatch=True),
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs",), blacklist_characters="\x00"
                ),
                max_size=30,
            ),
        ),
        max_size=5,
    )
)
def test_command_line_gives_back_any_pairs_through_shlex(
    pairs: list[tuple[str, str]],
) -> None:
    items = [f"{key}={value}" for key, value in pairs]
    line = command_line(
        "scan", items, profile="demo", run_doc=Path("/r/run.json"), dry_run=False
    )
    words = shlex.split(line)
    taken = [words[i + 1] for i, word in enumerate(words) if word == "--param"]
    assert taken == items
