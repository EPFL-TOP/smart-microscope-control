"""The scrub of a dataset's run document (algorithms design §2.2, #565).

A run document reaches the lab's share and a collaborator's laptop: a model path,
an output file or an error that quotes a file must not carry the name of an
account or of a host. The tables below are the specification of the rule; the
account the paths are written for is ``alice``, and no output may hold it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smc.data.scrub import _INSIDE, _STARTS_ABSOLUTE, scrub_document, scrub_text
from smc.run import RunDocument, StageRecord, StageStatus

#: The three forms of the acceptance criteria and the paths the design lists, whole strings.
WHOLE = {
    r"C:\Users\alice\models\m.pt": "m.pt",
    "/home/alice/models/m.pt": "m.pt",
    r"\\server\alice\models\m.pt": "m.pt",
    "C:/Users/alice/m.pt": "m.pt",
    "/Users/alice/My Models/m.pt": "m.pt",
    "/Volumes/lab/alice/x.pt": "x.pt",
    r"D:\data\alice\x.pt": "x.pt",
    # A blank in the account's own name, and a root written in another case: a
    # whole string is one path whatever it holds (adversarial review of #565).
    r"C:\Users\Alice Smith\models\m.pt": "m.pt",
    r"\\server\Alice Smith\x.pt": "x.pt",
    r"c:\users\alice\m.pt": "m.pt",
    r"C:\USERS\alice\m.pt": "m.pt",
    "/USERS/alice/x.pt": "x.pt",
}

#: A path inside a sentence: each becomes its last component and the words stay.
SENTENCES = {
    r"cannot open C:\Users\alice\m.pt: not found": "cannot open m.pt: not found",
    "see /home/alice/x.tif and C:/Users/alice/y.tif.": "see x.tif and y.tif.",
    "run folder /mnt/lab/alice/run1: cells file x is missing": (
        "run folder run1: cells file x is missing"
    ),
    "path (/home/alice/x.tif), then": "path (x.tif), then",
}

#: A path with nothing below the account, the drive, the share or the root:
#: its last component would be the account.
TILDES = [
    "/home/alice",
    "/Users/alice/",
    r"C:\Users\alice",
    r"\\server\alice",
    "\\\\server\\alice\\",
    "/",
    "C:\\",
    "/Users",
    "/home/",
    # The account root is case-insensitive: a mutant that compares it as written
    # writes `alice` for these (Windows paths are often typed in lower case).
    r"c:\users\alice",
    "/USERS/alice",
    "/Home/alice/",
    r"C:\Users\Alice Smith",
]

#: Text that is not a path, and must come back as it went in.
NOT_PATHS = [
    "scan/abc/A1/A1_0003.tif",
    "cells/abc.json",
    "http://example.com/a/x.html",
    "ratio and/or 1/2 of A1/A2 at E7/single",
    "a/b/c",
    "~/models/m.pt",
    "x /y z",
    "",
    "µm",
    # Each character of the POSIX pattern's lookbehind keeps one of these whole:
    # a relative path (`.`), a host and a path (`:`), a word and a dash (`-`),
    # and of the drive pattern's, a digit before the letter.
    "./models/m.pt",
    "../data/alice/x.tif",
    "see ./data/run/x.pt",
    "host:/b/c",
    "a-/b/c",
    r"id3e:\x",
]


@pytest.mark.parametrize(("text", "scrubbed"), [*WHOLE.items(), *SENTENCES.items()])
def test_a_path_becomes_its_last_component(text: str, scrubbed: str) -> None:
    assert scrub_text(text) == scrubbed


@pytest.mark.parametrize("text", TILDES)
def test_a_path_with_nothing_below_the_account_becomes_a_tilde(text: str) -> None:
    # "The last component" alone would write `alice`, the very name the scrub removes.
    assert scrub_text(text) == "~"


@pytest.mark.parametrize("text", NOT_PATHS)
def test_text_that_is_not_a_path_is_left_alone(text: str) -> None:
    # `http://example.com/...` is the case of a drive pattern with no word
    # boundary: it matched `p:/` and rewrote the end of the word `http`.
    assert scrub_text(text) == text


@pytest.mark.parametrize(
    ("text", "as_scrubbed"),
    [
        # A `file:///` URL is not a path to the rule: the account stays in it.
        ("file:///Users/alice/x.tif", "file:///Users/alice/x.tif"),
        # `/media/<name>` is a path of two components and its last one is the
        # name: the rule has no way to know a mount point from a folder of
        # files, so the name survives whole or in a sentence.
        ("/media/alice", "alice"),
        ("see /media/alice here", "see alice here"),
        # A blank inside a folder name, in a sentence: the path inside the text
        # runs to the next blank, so the rest of the name stays. A whole string
        # is not affected (`C:\Users\Alice Smith\x` is in WHOLE). Found by the
        # adversarial review of #565, and not in design §2.2's list of gaps.
        (r"run folder C:\Users\Jane Doe\smc\run1", r"run folder ~ Doe\smc\run1"),
        (
            r"cannot open \\labserver\home dirs\alice\m.pt: not found",
            r"cannot open ~ dirs\alice\m.pt: not found",
        ),
    ],
)
def test_the_known_gaps_are_pinned(text: str, as_scrubbed: str) -> None:
    # Known gaps, design §2.2: whoever closes one changes this test on purpose,
    # and the pattern that closes it needs a table of its own.
    assert scrub_text(text) == as_scrubbed


def test_the_drive_pattern_does_not_eat_the_end_of_a_word() -> None:
    # The first draft of the pattern turned `file:///Users/alice/x.tif` into
    # `filx.tif`, because `e:/` looked like a drive.
    assert _INSIDE.search("file:///x") is None
    assert _INSIDE.search("a C:/x") is not None
    assert _INSIDE.search("(C:\\x)") is not None
    assert _STARTS_ABSOLUTE.match("C:/x") is not None
    assert _STARTS_ABSOLUTE.match("relative/x") is None


@pytest.mark.parametrize(
    "text",
    [
        *WHOLE,
        *SENTENCES,
        *TILDES,
        *NOT_PATHS,
        "file:///Users/alice/x.tif",
        "/media/alice",
    ],
)
def test_scrubbing_twice_changes_nothing_the_second_time(text: str) -> None:
    once = scrub_text(text)
    assert scrub_text(once) == once


_PIECES = st.sampled_from(
    [
        "/",
        "\\",
        "C:",
        "e:",
        ":",
        "Users",
        "home",
        "alice",
        "x.tif",
        " ",
        ".",
        "~",
        "'",
        '"',
        "a",
        "1",
        "\\\\",
        "//",
        "(",
        ")",
        ",",
    ]
)


@given(st.lists(_PIECES, max_size=14).map("".join))
def test_scrubbing_is_idempotent_on_any_text_made_of_path_pieces(text: str) -> None:
    # The rule's own promise: exporting a dataset rewrites nothing
    # (`scrubbed` is empty). A property over pieces of paths and of words, which
    # a table of chosen strings cannot cover.
    once = scrub_text(text)
    assert scrub_text(once) == once


@given(st.text(alphabet=st.characters(blacklist_characters="/\\")))
def test_a_text_with_no_separator_is_never_changed(text: str) -> None:
    # A path needs a slash or a backslash: prose, labels and numbers that have
    # none are never touched, whatever they hold.
    assert scrub_text(text) == text


# --- the document ----------------------------------------------------------------


def _record(plugin: str, provides: str, **fields: Any) -> StageRecord:
    then = datetime(2026, 11, 2, 9, 30, tzinfo=timezone.utc)
    values: dict[str, Any] = {
        "plugin": plugin,
        "provides": provides,
        "attempt": "a1",
        "status": StageStatus.done,
        "started_utc": then,
        "finished_utc": then,
        "profile": "demo",
        "dry_run": False,
        "smc_version": "0",
        "params": {},
    }
    values.update(fields)
    return StageRecord(**values)


def _document(*records: StageRecord) -> RunDocument:
    return RunDocument(
        created_utc=datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc),
        stages={r.plugin: r for r in records},
    )


def _alice_document() -> RunDocument:
    return _document(
        _record(
            "scan",
            "scan",
            params={
                "out": r"C:\Users\alice\models\m.pt",
                "note": "/home/alice/models/m.pt",
                "a/b~c": r"\\server\alice\models\m.pt",
                "n": 3,
                "fine": "scan/t/A1_0000.tif",
            },
            result={"assumed": [r"C:\Users\alice\m.pt", "a plain line"]},
            error="cannot open /home/alice/x.tif: not found",
        ),
        _record(
            "cells",
            "cells",
            checkpoint={"out": "/home/alice/cells/t.json"},
        ),
    )


def test_scrub_document_lists_the_pointers_of_the_rewritten_values_in_order() -> None:
    document, pointers = scrub_document(_alice_document())

    assert pointers == (
        "/stages/scan/params/out",
        "/stages/scan/params/note",
        "/stages/scan/params/a~1b~0c",
        "/stages/scan/result/assumed/0",
        "/stages/scan/error",
        "/stages/cells/checkpoint/out",
    )
    assert "alice" not in json.dumps(document.model_dump(mode="json"))
    scan = document.stages["scan"]
    assert scan.params["out"] == "m.pt"
    assert scan.params["a/b~c"] == "m.pt"
    assert scan.params["n"] == 3
    assert scan.params["fine"] == "scan/t/A1_0000.tif"
    assert scan.result == {"assumed": ["m.pt", "a plain line"]}
    assert scan.error == "cannot open x.tif: not found"
    assert document.stages["cells"].checkpoint == {"out": "t.json"}


def test_a_failed_stage_lists_its_checkpoint_before_its_error() -> None:
    # The order of the manifest's `scrubbed` is a contract (params, result,
    # checkpoint, error): a failed `cells` stage has both a checkpoint and an
    # error that quote a path, and no other stage of the document does.
    document = _document(
        _record(
            "cells",
            "cells",
            status=StageStatus.failed,
            checkpoint={"out": "/home/alice/cells/t.json"},
            error="cannot open /home/alice/x.tif: not found",
        )
    )
    _, pointers = scrub_document(document)
    assert pointers == ("/stages/cells/checkpoint/out", "/stages/cells/error")


def test_a_stage_name_with_a_slash_is_escaped_in_its_pointer() -> None:
    document = _document(_record("odd/name~x", "scan", error="/home/alice/x.tif"))
    _, pointers = scrub_document(document)
    assert pointers == ("/stages/odd~1name~0x/error",)


def test_scrub_document_keeps_every_field_it_does_not_scrub() -> None:
    # FM-107: no record is rebuilt field by field, so a field that a later issue
    # adds to the record is carried too. The loop is over the model's own list.
    source = _document(
        _record(
            "scan",
            "scan",
            params={"out": "/home/alice/x"},
            preset="mine",
            offline=True,
            reads={"scan-plan": {"plugin": "scan-plan", "attempt": "a1"}},
            plugin_version="2.1",
        ),
        _record("cells", "cells", status=StageStatus.failed, error="boom"),
    )
    scrubbed, pointers = scrub_document(source)

    assert pointers == ("/stages/scan/params/out",)
    assert scrubbed.created_utc == source.created_utc
    assert scrubbed.schema_version == source.schema_version
    assert list(scrubbed.stages) == list(source.stages)
    for name, before in source.stages.items():
        after = scrubbed.stages[name]
        for field in StageRecord.model_fields:
            if field in ("params", "result", "checkpoint", "error"):
                continue
            assert getattr(after, field) == getattr(before, field), (name, field)
    assert scrubbed.stages["scan"].preset == "mine"
    assert scrubbed.stages["scan"].offline is True


def test_scrub_document_of_a_clean_document_is_the_same_document() -> None:
    clean = _document(
        _record("scan", "scan", params={"a": 1, "b": ["x", "y/z"]}, result={"k": None}),
        _record("cells", "cells", error="plain"),
    )
    scrubbed, pointers = scrub_document(clean)
    assert scrubbed == clean
    assert pointers == ()


def test_scrubbing_a_document_twice_rewrites_nothing_the_second_time() -> None:
    once, first = scrub_document(_alice_document())
    twice, second = scrub_document(once)
    assert first != ()
    assert second == ()
    assert twice == once


def test_a_key_is_never_rewritten() -> None:
    # No record has a path in a key today; if one ever does, the dataset carries
    # it, and the README and the Report say what the rule rewrites.
    document = _document(_record("scan", "scan", params={"/home/alice/x": "v"}))
    scrubbed, pointers = scrub_document(document)
    assert pointers == ()
    assert scrubbed.stages["scan"].params == {"/home/alice/x": "v"}


def test_the_pattern_constants_are_compiled_regular_expressions() -> None:
    assert isinstance(_INSIDE, re.Pattern)
    assert isinstance(_STARTS_ABSOLUTE, re.Pattern)
