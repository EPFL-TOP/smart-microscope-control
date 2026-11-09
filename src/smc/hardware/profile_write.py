"""Writing the profile's ``[objectives]`` as one block between two marker lines (m3 §20.7).

A profile is a file people maintain by hand and keep in git. ``smc objective
offset --write`` puts what ``objective-offsets`` measured into it, and the only
way to do that without rewriting someone's file is to own a part of it:

* **One block, and no line outside it is edited** (FM-107). It is appended when
  there is none and replaced in place when there is one, between ``BLOCK_BEGIN``
  and ``BLOCK_END``. A block that is damaged, or an ``[objectives]`` table
  outside it, is refused: smc does not guess which of two statements the
  operator meant.
* **The file's own shape is kept.** CRLF stays CRLF (a profile saved by Notepad),
  and a byte-order mark is dropped, which the loader reads the same (FM-80).
* **What is written is read back first** (FM-156): the new text must parse and
  validate as the loader reads it, and its ``[objectives]`` must be the offsets
  that were meant, else nothing is returned to write.

The caller writes the text (``write_atomically``, FM-38) and prints what was
replaced first (FM-172); this module only builds it.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import tomli_w
from pydantic import ValidationError

from smc.hardware.errors import ProfileError
from smc.hardware.profile import ObjectivesSection, Profile

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

__all__ = ["BLOCK_BEGIN", "BLOCK_END", "objectives_block", "with_objectives"]

BLOCK_BEGIN = "# >>> [objectives], written by smc objective offset --write (m2 §13)"
BLOCK_END = "# <<< [objectives], written by smc objective offset --write"
_BLOCK_NOTE = (
    "# The next write replaces everything up to the closing marker; edit values "
    "here only if no write follows."
)
#: A line with its own line end: ``\n`` or ``\r\n``, and a last line that has none.
#: Not ``splitlines``, which also splits on the form feed and U+2028, and would
#: find lines in a label that TOML keeps whole.
_LINE = re.compile(r"[^\n]*\n|[^\n]+")


def objectives_block(section: ObjectivesSection, *, newline: str = "\n") -> str:
    """The block for ``section``: the markers, a note, and the table, every line ending with ``newline``.

    A value that is ``None`` (not measured) is left out, never written as zero.
    The table is written by ``tomli_w``, which escapes a label's quotes and
    backslashes (FM-190); its lines are separated at an LF only, so that a
    character it leaves as it is (U+2028) is not taken for a line end.
    """
    table: dict[str, Any] = {"reference": section.reference}
    offsets: dict[str, dict[str, Any]] = {}
    for label, entry in section.offsets.items():
        values: dict[str, Any] = {}
        if entry.xy_offset_um is not None:
            values["xy_offset_um"] = list(entry.xy_offset_um)
        if entry.z_offset_um is not None:
            values["z_offset_um"] = entry.z_offset_um
        values["measured"] = entry.measured
        offsets[label] = values
    if offsets:
        table["offsets"] = offsets
    body = tomli_w.dumps({"objectives": table}).rstrip("\n")
    lines = [BLOCK_BEGIN, _BLOCK_NOTE, *body.split("\n"), BLOCK_END]
    return "".join(line + newline for line in lines)


def with_objectives(text: str, section: ObjectivesSection, *, where: str) -> str:
    """``text`` with its ``[objectives]`` block appended or replaced by ``section``'s; ``where`` names the file in every refusal.

    The lines that equal a marker once their line end is removed are the
    block's: a marker someone edited is a line of the profile. None of either
    and the block is appended after one empty line (a final line end is added
    first if the text has none); one opening followed by one closing and the
    lines from the first to the second are replaced. The line end is CRLF when
    the text holds one, else LF.

    Raises:
        ProfileError: The block is damaged (an opening without a closing marker,
            or more than one of either); an ``[objectives]`` table lies outside
            it; the text outside it is not TOML; or the new text does not read
            back as ``section``. Nothing is written by this function either way.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = _LINE.findall(text)
    begins = [i for i, line in enumerate(lines) if _bare(line) == BLOCK_BEGIN]
    ends = [i for i, line in enumerate(lines) if _bare(line) == BLOCK_END]

    if not begins and not ends:
        outside = text
    elif len(begins) == 1 and len(ends) == 1 and begins[0] < ends[0]:
        outside = "".join(lines[: begins[0]] + lines[ends[0] + 1 :])
    else:
        raise ProfileError(
            f"{where}: the [objectives] block that smc writes is damaged "
            f"({len(begins)} opening and {len(ends)} closing marker lines): restore "
            "it from git, or delete both markers and what lies between, then write "
            "again"
        )

    _refuse_a_table_outside(outside, where)
    block = objectives_block(section, newline=newline)
    if not begins:
        kept = text if not text or text.endswith("\n") else text + newline
        # One empty line between the profile and the block; none in an empty file.
        new = kept + (newline if kept else "") + block
    else:
        new = "".join(lines[: begins[0]]) + block + "".join(lines[ends[0] + 1 :])
    _refuse_what_does_not_read_back(new, section, where)
    return new


def _bare(line: str) -> str:
    """``line`` without its line end: the LF, and then the CR before it."""
    return line.removesuffix("\n").removesuffix("\r")


def _refuse_a_table_outside(outside: str, where: str) -> None:
    """Refuse a profile whose text outside the block has an ``[objectives]`` of its own.

    Two statements of the same offsets would contradict each other, and which
    one the operator meant is not for this function to guess. A dotted key at
    the root (``objectives.reference = ...``) is the same table.
    """
    problem = ""
    try:
        data = tomllib.loads(outside)
    except tomllib.TOMLDecodeError as exc:
        problem = (
            f"{where}: the profile is not valid TOML outside the block smc writes: "
            f"{exc}"
        )
    else:
        if "objectives" in data:
            problem = (
                f"{where}: the profile has an [objectives] table outside the block "
                "smc writes: delete it (its values are printed above) or edit it by "
                "hand instead of writing"
            )
    if problem:
        # Raised outside the handler: nothing here keeps the parser's error (FM-98).
        raise ProfileError(problem)


def _refuse_what_does_not_read_back(
    new: str, section: ObjectivesSection, where: str
) -> None:
    """Read ``new`` the way the loader does: it must be the profile it was, with ``section`` as its offsets (FM-156)."""
    read_back: ObjectivesSection | None = None
    try:
        read_back = Profile.model_validate(tomllib.loads(new)).objectives
    except (tomllib.TOMLDecodeError, ValidationError):
        read_back = None
    if read_back != section:
        raise ProfileError(
            f"{where}: the text written for [objectives] does not read back as the "
            "offsets; nothing was written"
        )
