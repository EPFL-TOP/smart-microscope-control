"""Take the account and host names out of a run document before it leaves the lab (algorithms design §2.2).

A run document is attached to issues and zipped onto the lab's share, and the
names of an account or a host reach it only through paths: a model path in a
parameter, an output file, an error that quotes a file. This module rewrites
those paths to their last component, in the four fields where such text lives,
and says which values it rewrote. It is pure (no file, no clock) and idempotent:
a dataset that was scrubbed once is not rewritten by an export of it.

The rule is the design's, as corrected while planning #565. Two first drafts of
it failed on measured text and are the reason for the patterns below: a drive
pattern with no word boundary turned ``file:///Users/alice/x.tif`` into
``filx.tif`` (it matched ``e:/``), and a rule keyed on ``/Users`` and ``/home``
left ``/mnt/lab/alice/run1`` alone in the ``error`` of a failed ``cells`` stage,
which quotes the run folder.
"""

from __future__ import annotations

import re
from typing import Any

from smc.run import RunDocument

__all__ = ["scrub_document", "scrub_text"]

#: A string that starts like an absolute path *is* a path: it becomes its last
#: component whole, spaces and all (``/Users/alice/My Models/m.pt``).
_STARTS_ABSOLUTE = re.compile(r"/|[A-Za-z]:[\\/]|\\\\")

#: A path inside any other text: a drive that is not the end of a word (so that
#: ``file:///`` and ``http://`` are not drives), a UNC path, and a POSIX path of
#: two components or more that does not continue a word, a URL or a relative
#: path (``a/b/c``, ``~/x`` and ``ratio and/or`` stay as they are). Each runs
#: to the next blank or quote. Known gap, pinned in
#: ``test_the_known_gaps_are_pinned``: a folder name with a blank in it, in a
#: sentence (``C:\Users\Jane Doe\x``), is cut at the blank and the rest of the
#: name stays. Design §2.2 does not list it (it lists ``file:///`` and
#: ``/media/<name>``); the plan deviation on #565 asks the owner whether to
#: accept it or to close it in the rule.
_INSIDE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s'\"]*"
    r"|\\\\[^\s'\"]*"
    r"|(?<![\w.~:/-])/[^\s'\"/]+/[^\s'\"]*"
)

_SEPARATORS = re.compile(r"[\\/]+")
_DRIVE = re.compile(r"[A-Za-z]:[\\/]")

#: The first folder of an account's tree: what is below it by one component is
#: the account itself, so a path that stops there says ``~`` and not the name.
_ACCOUNT_ROOTS = frozenset({"users", "home"})

#: The fields of a stage record that hold what a plugin or the layer wrote as
#: text; every other field is a name, a time, a status or a version.
_SCRUBBED_FIELDS = ("params", "result", "checkpoint", "error")


def _leaf(path: str) -> str:
    r"""The last component of ``path``, or ``~`` when nothing is below the account, the drive or the share.

    ``floor`` is the number of components that are the account's, not the
    file's: ``Users/<name>`` and ``home/<name>`` have two, and so has a UNC
    path (``\\host\share``). A path that has no more components than that
    would give the account (or the share) as its last component, which is what
    the scrub is there to remove.
    """
    parts = [part for part in _SEPARATORS.split(path) if part]
    if _DRIVE.match(path):
        parts = parts[1:]
    is_account_tree = bool(parts) and parts[0].lower() in _ACCOUNT_ROOTS
    floor = 2 if path.startswith("\\\\") or is_account_tree else 0
    return parts[-1] if len(parts) > floor else "~"


def scrub_text(text: str) -> str:
    """``text`` with every absolute path in it reduced to its last component.

    A string that starts like an absolute path is one path, and is replaced as
    a whole; in any other string each match of the three patterns is. Relative
    paths, URLs, ratios and prose come back as they went in.
    """
    if _STARTS_ABSOLUTE.match(text):
        return _leaf(text)
    return _INSIDE.sub(lambda found: _leaf(found.group(0)), text)


def _pointer(parts: list[str]) -> str:
    """An RFC 6901 JSON pointer: ``~`` is ``~0`` and ``/`` is ``~1`` in a part (``~`` first)."""
    return "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in parts)


def _walk(value: Any, where: list[str], pointers: list[str]) -> Any:
    """``value`` with its string values scrubbed, depth first, in key and list order.

    A key is never rewritten: no record holds a path in one today (they are
    slots, wells, labels, plugin names), and a rule that rewrote keys could
    merge two of them.
    """
    if isinstance(value, str):
        scrubbed = scrub_text(value)
        if scrubbed != value:
            pointers.append(_pointer(where))
        return scrubbed
    if isinstance(value, dict):
        return {
            key: _walk(item, [*where, str(key)], pointers)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _walk(item, [*where, str(index)], pointers)
            for index, item in enumerate(value)
        ]
    return value


def scrub_document(document: RunDocument) -> tuple[RunDocument, tuple[str, ...]]:
    """The document with the paths of its records scrubbed, and the pointers of the values it rewrote.

    The records are carried as they dump and only the four text fields are
    walked, so a field that a later version adds to the record travels with it
    (FM-107: nothing is rebuilt field by field). The result is validated as a
    document before it is returned, so a scrub that broke the document fails
    here and not in the reader. The pointers go into the manifest, in the
    order of the file: the stages as it has them, and in a record ``params``,
    ``result``, ``checkpoint`` then ``error``.
    """
    dumped = document.model_dump(mode="json")
    pointers: list[str] = []
    for stage, record in dumped["stages"].items():
        for field in _SCRUBBED_FIELDS:
            if field in record:
                record[field] = _walk(
                    record[field], ["stages", str(stage), field], pointers
                )
    return RunDocument.model_validate(dumped), tuple(pointers)
