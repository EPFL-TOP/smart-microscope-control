"""The run document (m3 §1.7): what is written, what is refused, what is stale.

Every document lives under ``tmp_path``. The exact text of a refusal is asserted
(FM-45): a bare word would also match the path, which is named after the test.
"""

from __future__ import annotations

import codecs
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from smc.run import (
    SCHEMA_VERSION,
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)
from smc.run import document as document_mod

T0 = datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)
DONE = StageStatus.done


def _record(
    plugin: str = "a",
    *,
    provides: str | None = None,
    minute: int = 0,
    status: StageStatus = DONE,
    attempt: str | None = None,
    profile: str = "demo",
    dry_run: bool = False,
    reads: dict[str, ReadRecord | None] | None = None,
    result: dict[str, Any] | None = None,
) -> StageRecord:
    started = T0 + timedelta(minutes=minute)
    return StageRecord(
        plugin=plugin,
        provides=provides or plugin,
        attempt=attempt or f"{plugin}-{minute}",
        status=status,
        started_utc=started,
        finished_utc=started + timedelta(seconds=1) if status is DONE else None,
        profile=profile,
        dry_run=dry_run,
        smc_version="0.0.1",
        params={},
        result=result if result is not None else ({} if status is DONE else None),
        reads=reads or {},
    )


def _read(of: StageRecord) -> ReadRecord:
    return ReadRecord(plugin=of.plugin, attempt=of.attempt)


def _doc(*records: StageRecord) -> RunDocument:
    return RunDocument(
        created_utc=T0, stages={record.plugin: record for record in records}
    )


def _write_json(path: Path, value: object) -> bytes:
    data = json.dumps(value).encode("utf-8")
    path.write_bytes(data)
    return data


def _valid_json(**top: object) -> dict[str, Any]:
    document = {"schema_version": SCHEMA_VERSION, "created_utc": "2026-10-26T09:00:00Z"}
    document.update(top)
    return document


def _error_of(path: Path) -> RunDocumentError:
    with pytest.raises(RunDocumentError) as raised:
        RunDocument.load(path)
    return raised.value


# --- round trip and encoding -------------------------------------------------


def test_a_written_document_loads_back_equal(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    written = write_stage(path, _record("a", result={"x_um": 1.5, "n": None}))
    loaded = RunDocument.load(path)
    assert loaded == written
    assert loaded.schema_version == SCHEMA_VERSION == 1
    assert loaded.stages["a"].result == {"x_um": 1.5, "n": None}
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert '\n  "schema_version": 1,' in text  # indent=2


def test_a_lone_surrogate_in_a_result_is_written_as_ascii_and_reads_back(
    tmp_path: Path,
) -> None:
    # FM-05: an adapter's string can hold a lone surrogate, and "µm" is not ASCII.
    path = tmp_path / "run.json"
    label = f"{chr(0xDC80)} µm"
    written = write_stage(path, _record("a", result={"label": label}))
    assert max(path.read_bytes()) < 0x80
    loaded = RunDocument.load(path)
    assert loaded == written
    result = loaded.stages["a"].result
    assert result is not None
    assert result["label"] == label


def test_a_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    # FM-80: Notepad's "UTF-8" before Windows 10 1903 starts the file with a mark.
    path = tmp_path / "run.json"
    written = write_stage(path, _record("a"))
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes())
    assert RunDocument.load(path) == written


def test_a_missing_file_gives_a_new_document_and_creates_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.json"
    doc = RunDocument.load_or_new(path)
    assert doc.stages == {}
    assert doc.schema_version == SCHEMA_VERSION
    assert doc.created_utc.utcoffset() == timedelta(0)
    assert list(tmp_path.iterdir()) == []


def test_a_missing_file_is_refused_by_load(tmp_path: Path) -> None:
    error = _error_of(tmp_path / "run.json")
    assert error.reason == "does not exist"
    assert error.path == tmp_path / "run.json"


# --- refused documents -------------------------------------------------------


def test_a_newer_schema_version_is_refused_and_names_both_versions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.json"
    _write_json(path, _valid_json(schema_version=2))
    assert _error_of(path).reason == (
        "written by a newer smc (schema 2); this smc reads schema 1 — upgrade smc"
    )


def test_an_unknown_field_is_refused_not_dropped(tmp_path: Path) -> None:
    # FM-107: loading and rewriting would drop a field this smc does not know.
    path = tmp_path / "run.json"
    record = json.loads(_record("a").model_dump_json())
    record["future_field"] = 1
    _write_json(path, _valid_json(stages={"a": record}))
    reason = _error_of(path).reason
    assert reason.startswith("stages.a.future_field: Extra inputs are not permitted")
    assert "upgrade smc" in reason


def test_invalid_json_is_refused_with_its_line(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{\n  "schema_version": 1,\n  "stages": {,\n}\n', encoding="utf-8")
    assert _error_of(path).reason == (
        "is not valid JSON: Expecting property name enclosed in double quotes "
        "(line 3, column 14); it was left as it is, restore it from a copy"
    )


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        pytest.param(
            b"[1, 2]", "holds a JSON list, not an object", id="top-level-not-an-object"
        ),
        pytest.param(
            b'{"created_utc": "2026-10-26T09:00:00Z"}',
            "has no schema_version; smc always writes one, so this file is not a "
            "run document",
            id="no-schema-version",
        ),
        pytest.param(
            b'{"schema_version": "1"}',
            "schema_version must be an integer, got '1'",
            id="schema-version-text",
        ),
        pytest.param(
            b'{"schema_version": true}',
            "schema_version must be an integer, got True",
            id="schema-version-bool",
        ),
        pytest.param(
            b'{"schema_version": 0}',
            "schema_version must be 1 or more, got 0",
            id="schema-version-zero",
        ),
        pytest.param(
            b'{"schema_version": 1, "created_utc": NaN}',
            "contains NaN, which JSON does not allow",
            id="nan",
        ),
        pytest.param(
            b'{"schema_version": 1}',
            "created_utc: Field required",
            id="validation",
        ),
        pytest.param(
            b"\xff\xfe{\x00}\x00",
            "is not UTF-8 text (invalid start byte at byte 0)",
            id="not-utf-8",
        ),
    ],
)
def test_a_document_that_cannot_be_read_is_refused_with_its_reason(
    tmp_path: Path, content: bytes, reason: str
) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(content)
    error = _error_of(path)
    assert error.reason.startswith(reason)
    assert str(error).startswith(f"{path}: ")
    assert path.read_bytes() == content


def test_a_folder_is_refused_as_a_run_document(tmp_path: Path) -> None:
    error = _error_of(tmp_path)
    assert error.reason == "is a folder, not a run document; name a file"


def test_an_unreadable_document_is_never_overwritten(tmp_path: Path) -> None:
    # The document may be the only record of an experiment (m3 §1.7).
    path = tmp_path / "run.json"
    corrupt = b'{"schema_version": 1, "stages": '
    path.write_bytes(corrupt)
    with pytest.raises(RunDocumentError) as raised:
        write_stage(path, _record("a"))
    assert raised.value.reason.startswith("is not valid JSON: ")
    assert raised.value.record == _record("a")  # the work is not lost with it
    assert path.read_bytes() == corrupt
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    with pytest.raises(RunDocumentError):
        RunDocument.load_or_new(path)


def test_a_run_document_error_survives_pickle() -> None:
    # Like every error of the layer: it may cross a worker process.
    record = _record("a")
    error = RunDocumentError(
        Path("run.json"), "could not be replaced", kept=Path(".tmp"), record=record
    )
    copy = pickle.loads(pickle.dumps(error))
    assert (copy.path, copy.reason, copy.kept, copy.record) == (
        Path("run.json"),
        "could not be replaced",
        Path(".tmp"),
        record,
    )
    assert str(copy) == str(error)


# --- writing -----------------------------------------------------------------


def test_write_stage_keeps_the_other_stages_written_meanwhile(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    write_stage(path, _record("a", minute=0))
    write_stage(path, _record("b", minute=1))  # another writer, in between
    written = write_stage(path, _record("a", minute=2))
    assert sorted(written.stages) == ["a", "b"]
    assert written.stages["a"].attempt == "a-2"
    assert RunDocument.load(path) == written


def test_a_failed_replace_keeps_the_document_and_names_the_kept_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-38: the old file stays whole; the new text is on disk and named.
    path = tmp_path / "run.json"
    write_stage(path, _record("a", minute=0))
    before = path.read_bytes()

    def refuse(src: object, dst: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(document_mod.os, "replace", refuse)
    monkeypatch.setattr(document_mod.time, "sleep", lambda seconds: None)
    new = _record("b", minute=1)
    with pytest.raises(RunDocumentError) as raised:
        write_stage(path, new)
    error = raised.value
    assert path.read_bytes() == before
    assert error.kept is not None
    assert error.kept.parent == tmp_path
    assert error.kept.name.startswith(".run.json.")
    assert error.kept.name.endswith(".smc-tmp")
    assert error.reason == (
        f"could not be replaced (Permission denied); it is as it was, and the new "
        f"text is kept as {error.kept}. Close any program that holds run.json "
        f"open, then run the stage again"
    )
    assert error.record == new
    kept = json.loads(error.kept.read_text(encoding="utf-8"))
    assert sorted(kept["stages"]) == ["a", "b"]


def test_a_replace_refused_four_times_then_allowed_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FM-24: antivirus or an indexer holds the file for a moment on Windows.
    path = tmp_path / "run.json"
    real_replace, refused, waits = os.replace, [], []

    def flaky(src: Any, dst: Any) -> None:
        if len(refused) < 4:
            refused.append(src)
            raise PermissionError(13, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(document_mod.os, "replace", flaky)
    monkeypatch.setattr(document_mod.time, "sleep", waits.append)
    written = write_stage(path, _record("a"))
    assert len(refused) == 4
    assert waits == [0.05] * 4
    assert RunDocument.load(path) == written
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_a_failed_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    write_stage(path, _record("a", minute=0))
    before = path.read_bytes()

    def full(fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(document_mod.os, "fsync", full)
    with pytest.raises(RunDocumentError) as raised:
        write_stage(path, _record("b", minute=1))
    assert raised.value.reason == (
        "could not be written (No space left on device); it is as it was"
    )
    assert raised.value.kept is None
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_a_folder_that_does_not_exist_is_a_run_document_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing" / "run.json"
    with pytest.raises(RunDocumentError) as raised:
        write_stage(path, _record("a"))
    assert raised.value.reason.startswith("could not be written (")
    assert raised.value.kept is None


def test_a_document_is_created_with_the_mode_of_a_plain_write(tmp_path: Path) -> None:
    # `mkstemp`'s 0600 would survive the rename (review of #89).
    path = tmp_path / "run.json"
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    write_stage(path, _record("a"))
    assert path.stat().st_mode & 0o777 == plain.stat().st_mode & 0o777


# --- slots and staleness -----------------------------------------------------


def test_provider_is_the_latest_attempt_even_when_it_failed() -> None:
    older = _record("register", provides="plate", minute=0)
    failed = _record("register", provides="plate", minute=5, status=StageStatus.failed)
    doc = _doc(failed)  # one record per plugin: the re-run replaced the older one
    assert older.attempt != failed.attempt
    provider = doc.provider("plate")
    assert provider is not None
    assert (provider.attempt, provider.status) == ("register-5", StageStatus.failed)
    assert doc.provider("nothing") is None


def test_two_plugins_providing_one_slot_resolve_to_the_latest() -> None:
    registered = _record("register", provides="plate", minute=1)
    found = _record("find", provides="plate", minute=2)
    assert _doc(registered, found).provider("plate") == found
    registered_later = _record("register", provides="plate", minute=3)
    assert _doc(registered_later, found).provider("plate") == registered_later


def test_a_tie_on_the_start_time_goes_to_the_plugin_name() -> None:
    a = _record("a-plate", provides="plate", minute=1)
    b = _record("b-plate", provides="plate", minute=1)
    assert _doc(a, b).provider("plate") == b
    assert _doc(b, a).provider("plate") == b


def test_rerunning_an_upstream_makes_its_reader_stale() -> None:
    up = _record("up", minute=0)
    reader = _record("reader", minute=1, reads={"up": _read(up)})
    assert not _doc(up, reader).is_stale("reader")
    rerun = _record("up", minute=2, attempt="up-again")
    assert _doc(rerun, reader).is_stale("reader")
    assert not _doc(rerun, reader).is_stale("up")  # it read nothing


def test_a_reader_whose_upstream_vanished_is_stale() -> None:
    up = _record("up", minute=0)
    reader = _record("reader", minute=1, reads={"up": _read(up)})
    assert _doc(reader).is_stale("reader")


def test_a_provider_by_another_plugin_makes_the_reader_stale() -> None:
    registered = _record("register", provides="plate", minute=0)
    reader = _record("wells", minute=1, reads={"plate": _read(registered)})
    found = _record("find", provides="plate", minute=2)
    assert _doc(registered, found, reader).is_stale("wells")


def test_staleness_propagates_through_two_levels() -> None:
    a = _record("a", minute=0)
    b = _record("b", minute=1, reads={"a": _read(a)})
    c = _record("c", minute=2, reads={"b": _read(b)})
    assert not _doc(a, b, c).is_stale("c")
    a_again = _record("a", minute=3, attempt="a-again")
    doc = _doc(a_again, b, c)
    assert doc.is_stale("b")
    assert doc.is_stale("c")  # its own input b is current, but b is stale


def test_an_unknown_plugin_and_a_cycle_are_not_stale() -> None:
    assert not _doc().is_stale("nobody")
    # Hand-edited: a and b read each other; a visited set ends the walk.
    a = _record("a", minute=0, attempt="a1")
    b = _record("b", minute=1, attempt="b1")
    a = a.model_copy(update={"reads": {"b": _read(b)}})
    b = b.model_copy(update={"reads": {"a": _read(a)}})
    assert not _doc(a, b).is_stale("a")


def test_an_optional_read_recorded_as_none_is_stale_once_provided() -> None:
    scan = _record("scan", minute=0, reads={"timings": None})
    assert not _doc(scan).is_stale("scan")
    timing = _record("timing", provides="timings", minute=1)
    assert _doc(scan, timing).is_stale("scan")
    unfinished = _record(
        "timing", provides="timings", minute=1, status=StageStatus.failed
    )
    assert not _doc(scan, unfinished).is_stale("scan")  # still not usable


# --- usability ---------------------------------------------------------------


def _unusable(doc: RunDocument, slot: str = "plate", **kw: Any) -> str | None:
    return doc.unusable(
        slot, profile=kw.get("profile", "demo"), dry_run=kw.get("dry_run", False)
    )


def test_a_usable_slot_has_no_reason() -> None:
    assert _unusable(_doc(_record("register", provides="plate"))) is None


@pytest.mark.parametrize(
    ("make", "kw", "reason"),
    [
        pytest.param(
            lambda: _doc(),
            {},
            "the run document has no plate; smc run --list shows which plugin "
            "provides it",
            id="no-provider",
        ),
        pytest.param(
            lambda: _doc(
                _record("register", provides="plate", status=StageStatus.failed)
            ),
            {},
            "the last run of register ended failed; run it again",
            id="failed",
        ),
        pytest.param(
            lambda: _doc(
                _record("register", provides="plate", status=StageStatus.stopped)
            ),
            {},
            "the last run of register ended stopped; run it again",
            id="stopped",
        ),
        pytest.param(
            lambda: _doc(
                _record("register", provides="plate", status=StageStatus.running)
            ),
            {},
            "the last run of register is still running or was interrupted; run it "
            "again",
            id="running",
        ),
        pytest.param(
            lambda: _stale_plate_document(),
            {},
            "register read its inputs from an earlier run; run it again",
            id="stale",
        ),
        pytest.param(
            lambda: _doc(_record("register", provides="plate", profile="other")),
            {},
            "register was run on other, this is demo; run register again on this "
            "profile",
            id="other-profile",
        ),
        pytest.param(
            lambda: _doc(_record("register", provides="plate", dry_run=True)),
            {},
            "register was a dry run; run it without --dry-run",
            id="dry-run-provider",
        ),
    ],
)
def test_unusable_reasons(make: Any, kw: dict[str, Any], reason: str) -> None:
    assert _unusable(make(), **kw) == reason


def _stale_plate_document() -> RunDocument:
    origin = _record("origin", minute=0)
    register = _record(
        "register", provides="plate", minute=1, reads={"origin": _read(origin)}
    )
    return _doc(_record("origin", minute=2, attempt="origin-again"), register)


def test_the_checks_run_in_the_order_of_the_table() -> None:
    # A provider that is failed, stale, on another profile and a dry run at
    # once is reported as failed: the first row that applies wins.
    origin = _record("origin", minute=0)
    register = _record(
        "register",
        provides="plate",
        minute=1,
        status=StageStatus.failed,
        profile="other",
        dry_run=True,
        reads={"origin": _read(origin)},
    )
    doc = _doc(_record("origin", minute=2, attempt="again"), register)
    assert _unusable(doc) == "the last run of register ended failed; run it again"


def test_a_dry_run_may_read_a_real_result() -> None:
    doc = _doc(_record("register", provides="plate", dry_run=False))
    assert _unusable(doc, dry_run=True) is None
    dry = _doc(_record("register", provides="plate", dry_run=True))
    assert _unusable(dry, dry_run=True) is None
    assert _unusable(dry, dry_run=False) == (
        "register was a dry run; run it without --dry-run"
    )
