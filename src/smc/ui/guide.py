"""What the plate page says to do next: a value and a command, no widget (m5 §8.3).

The guide is a function of the run document's checklist, the page's selection and
captures, and the profile; it keeps no state, and the page draws it above its tools.
It holds no Panel and no device, so every sentence it builds is tested as text. Its
one promise is the one the page's *Run* button keeps: the pairs the step shows are the
pairs it runs, and ``command_line`` is the same call typed in a terminal (principle 5).
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smc.ui.controller import CheckItem, LoadedPreset

__all__ = [
    "CAPTURE_WELLS",
    "DRY_RUN_NO_FRAME",
    "SELECT_WELLS",
    "NextStep",
    "command_line",
    "next_slot",
    "preset_differences",
    "preset_note_text",
]

#: What the scan plan waits for. The page cannot choose wells for the operator:
#: which wells to image is the experiment (m5 §8.6).
SELECT_WELLS = (
    "select the wells to scan: click them on the map, or type a range in Wells "
    "(A1:D6, C*, all)"
)

#: What plate registration waits for on a stand that has no sample to register from:
#: the plate's place is measured by centring wells, never guessed.
CAPTURE_WELLS = (
    "centre a well under the objective (Live and the Drive pad), click it on the "
    "map and press Capture centre; a second well, far from the first, fits the "
    "rotation too"
)

#: Why the step stops at ``scan`` on a page served with ``--dry-run`` (m5 §8.8.1): a dry
#: run walks the fields and takes no frame, so ``cells`` and what follows fail on a
#: missing frame, and the way out of a terminal run is not obvious (a run that is not a
#: dry run refuses what a dry run made).
DRY_RUN_NO_FRAME = (
    "this page is a dry run: scan walks the fields and takes no frame, so cells and "
    "the tools after it have nothing to read. Serve the page without --dry-run on a "
    "stand whose stage is simulated (demo-sample), or run the chain with smc run "
    "from plate-registration, without --dry-run: a run that is not a dry run "
    "refuses what a dry run made"
)


@dataclass(frozen=True, slots=True)
class NextStep:
    """The tool the run document needs next, with the values to run it and why.

    ``params`` are ``smc run --param`` pairs, the call the button makes. ``why``
    holds one line for the tool and one for each value it carries, each saying
    where the value comes from; the others are the other ways to fill the slot.
    ``plugin`` is ``None`` when there is nothing to run (the workflow is done, or
    no registered plugin fills the slot), and ``blocked`` is why the tool cannot run
    now: the step is still drawn, so the operator reads what to do. ``command`` is
    ``""`` without a tool.
    """

    plugin: str | None
    params: tuple[str, ...]
    why: tuple[str, ...]
    command: str
    blocked: str | None


def next_slot(checklist: Sequence[CheckItem]) -> CheckItem | None:
    """The first slot the workflow needs and does not have in a usable state, or ``None``.

    The optional slots (``camera-calibration``, ``timings``, a third party's) are
    never a next step: the workflow runs without them.
    """
    for item in checklist:
        if not item.optional and item.state != "ok":
            return item
    return None


def command_line(
    plugin: str,
    params: Sequence[str],
    *,
    profile: str,
    run_doc: Path,
    dry_run: bool,
    preset: str | None = None,
) -> str:
    """The ``smc run`` command that makes the call a step makes, quoted for a POSIX shell.

    ``shlex.join`` makes a JSON value one single-quoted word, so the operator
    copies it whole. How it pastes into PowerShell or ``cmd.exe`` is not verified
    (m5 §9.1); the run document's record of the run is the reference, whichever
    shell reads the line.

    ``preset`` is what ``smc run --preset`` takes (the page passes the file's
    path): given, the terminal's run applies the preset as the page's does, and
    its record names the preset as the page's would (m5 §10.5). The pairs that
    follow repeat what the preset gives, so the command says all it runs.
    """
    words = ["smc", "run", plugin, "-p", profile, "--run-doc", str(run_doc)]
    if dry_run:
        words.append("--dry-run")
    if preset is not None:
        words += ["--preset", preset]
    for pair in params:
        words += ["--param", pair]
    return shlex.join(words)


def preset_differences(
    expected: Mapping[str, Mapping[str, str]],
    actual: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    """What the page's fields hold that the loaded preset's load did not set, one line each (m5 §10.6).

    ``expected`` is what the load put in every field (the preset's value, else
    the profile's suggestion, else ``""``), per tool in the page's order, and
    ``actual`` what the fields hold now. A line is
    ``<tool> <key>: <value> (the preset: <expected>)`` for each field whose text,
    stripped, is not the expected one, with ``(default)`` for an empty text: the
    plugin's own default is what an empty field sends. A field the page no longer
    shows is read as empty. A value is shown on one line and cut, since a field
    may hold the whole of a JSON list; it is compared whole.
    """
    lines: list[str] = []
    for plugin, fields in expected.items():
        now = actual.get(plugin, {})
        for key, wanted in fields.items():
            found = now.get(key, "").strip()
            if found != wanted.strip():
                lines.append(
                    f"{plugin} {key}: {_shown(found)} (the preset: {_shown(wanted)})"
                )
    return tuple(lines)


def preset_note_text(loaded: LoadedPreset | None, differences: Sequence[str]) -> str:
    """The note under the *Preset* region: which preset, what it could not apply, what changed since.

    Plain text with line breaks, never markup (FM-139): the names come from a
    file a person edited. The mismatches and the differences are each cut at
    ``_NOTE_LINES`` lines, then ``… <n> more``, as a result card's lines are.
    """
    if loaded is None:
        return "no preset loaded"
    lines = [f"loaded: {loaded.file.name}  ({loaded.file.path})"]
    lines += [f"  {line}" for line in _cut_lines(loaded.mismatches)]
    if differences:
        lines.append("changed since loading:")
        lines += [f"  {line}" for line in _cut_lines(differences)]
    else:
        lines.append("the page holds the preset's values")
    return "\n".join(lines)


#: A note shows this many lines of each kind, as a result card shows its values (§2.4).
_NOTE_LINES = 8
#: A value in a difference is shown at most this long: a field may hold a whole JSON list.
_VALUE_CHARS = 80


def _cut_lines(lines: Sequence[str]) -> list[str]:
    if len(lines) <= _NOTE_LINES:
        return [_flat(line) for line in lines]
    return [
        *(_flat(line) for line in lines[:_NOTE_LINES]),
        f"… {len(lines) - _NOTE_LINES} more",
    ]


def _flat(text: str) -> str:
    """``text`` on one line: its line breaks are spaces."""
    return " ".join(text.split("\n")).replace("\r", " ")


def _shown(text: str) -> str:
    """A field's text as a difference shows it: ``(default)`` when empty, one line, cut."""
    if not text:
        return "(default)"
    flat = _flat(text)
    return flat if len(flat) <= _VALUE_CHARS else flat[:_VALUE_CHARS] + "…"
