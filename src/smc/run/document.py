"""The run document: one JSON file per experiment, written stage by stage (m3 §1.7).

Plate registration, plate finding, calibration and the scan are separate
commands run hours apart, and each one's result is the next one's input. The
document is where they meet, so it is held to the standard of a lab notebook:

* **It is never half-written.** Every write goes through a temporary file and
  ``os.replace`` (FM-38), and a replace that Windows refuses for a moment
  (antivirus, an indexer: FM-24) is retried before it is given up.
* **It is never overwritten when it cannot be read.** It may be the only record
  of an experiment; a corrupt, newer or hand-edited file is refused with its
  reason and left as it is, and ``extra="forbid"`` means a field this smc does
  not know is refused rather than loaded and rewritten without it (FM-107).
* **It says what is stale.** A stage records which attempt of which stage it
  read; once an upstream stage is run again, its readers are stale, and
  ``unusable`` says so before a downstream stage runs on an input that no
  longer matches (FM-106: a ``running``, ``stopped`` or ``failed`` record is
  never mistaken for a ``done`` one).
* **It is plain ASCII JSON.** A lone surrogate from an adapter (FM-05) and a
  ``µ`` are escaped, so no console or code page can mangle the file, and a
  byte-order mark from Notepad is accepted (FM-80).

Nothing here imports ``smc.plugins`` or ``smc.hardware``: plugins depend on the
document, not the reverse.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

__all__ = [
    "SCHEMA_VERSION",
    "ReadRecord",
    "RunDocument",
    "RunDocumentError",
    "StageRecord",
    "StageStatus",
    "write_stage",
]

#: Every change of the document's fields bumps this, and ``load`` migrates an
#: older version in code: an unknown field is refused (FM-107), so an older smc
#: cannot be left to drop what a newer one wrote.
SCHEMA_VERSION = 1

#: Windows refuses to replace a file that antivirus or a search indexer holds
#: open for a moment (``PermissionError``), and the file was written a moment
#: ago (FM-24): a few short waits, not a lost result. As ``_mm_child._write``.
_REPLACE_TRIES = 5
_REPLACE_WAIT_S = 0.05


class StageStatus(str, Enum):
    """How the last attempt of a stage ended; only ``done`` is a result to read."""

    running = "running"
    done = "done"
    stopped = "stopped"
    failed = "failed"


class ReadRecord(BaseModel):
    """Which attempt of which stage a stage read a slot from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin: str
    attempt: str


class StageRecord(BaseModel):
    """The record of one plugin's last attempt, keyed by the plugin's name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin: str
    provides: str
    attempt: str
    status: StageStatus
    started_utc: AwareDatetime
    finished_utc: AwareDatetime | None = None
    profile: str
    dry_run: bool
    smc_version: str
    plugin_version: str | None = None
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    reads: dict[str, ReadRecord | None] = {}
    error: str | None = None


class RunDocumentError(Exception):
    """The run document could not be read or written; ``reason`` says what to do.

    Not a ``HardwareError``: nothing is wrong with the stand. ``kept`` is the
    temporary file that holds the whole new text when only the final rename
    failed, and ``record`` the stage record that could not be written, so a
    caller can still show its result (FM-34).
    """

    def __init__(
        self,
        path: Path,
        reason: str,
        kept: Path | None = None,
        record: StageRecord | None = None,
    ) -> None:
        self.path = path
        self.reason = reason
        self.kept = kept
        self.record = record
        super().__init__(f"{path}: {reason}")

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuilt from the arguments, not the message, so the error survives a
        # worker process (pickle), as the errors of smc.hardware do.
        return (type(self), (self.path, self.reason, self.kept, self.record))


class RunDocument(BaseModel):
    """The whole document: a schema version, its creation time and one record per plugin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    created_utc: AwareDatetime
    stages: dict[str, StageRecord] = {}

    @classmethod
    def new(cls) -> RunDocument:
        """An empty document; nothing is written until a stage is."""
        return cls(created_utc=datetime.now(timezone.utc))

    @classmethod
    def load(cls, path: Path) -> RunDocument:
        """Read ``path``; never a partial document.

        Raises:
            RunDocumentError: The file is missing, a folder, unreadable, not
                UTF-8, not JSON, not an object, of a schema this smc does not
                read, or does not validate. The file is left as it is.
        """
        loaded = cls._read(path, missing_ok=False)
        assert loaded is not None  # missing_ok=False raises instead
        return loaded

    @classmethod
    def load_or_new(cls, path: Path) -> RunDocument:
        """``load``, except that a missing file gives a new document.

        Only a missing file: an unreadable one raises, because the document
        may be the only record of an experiment and the next write would
        replace it. The file is created by the first ``write_stage``, never
        here.
        """
        loaded = cls._read(path, missing_ok=True)
        return loaded if loaded is not None else cls.new()

    @classmethod
    def _read(cls, path: Path, *, missing_ok: bool) -> RunDocument | None:
        try:
            if path.is_dir():
                raise RunDocumentError(
                    path, "is a folder, not a run document; name a file"
                )
            data = path.read_bytes()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise RunDocumentError(path, "does not exist") from None
        except OSError as exc:
            raise RunDocumentError(
                path, f"could not be read ({exc.strerror or exc})"
            ) from exc
        return cls._parse(path, data)

    @classmethod
    def _parse(cls, path: Path, data: bytes) -> RunDocument:
        try:
            # utf-8-sig: a byte-order mark from Notepad is accepted (FM-80).
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RunDocumentError(
                path,
                f"is not UTF-8 text ({exc.reason} at byte {exc.start}); save it as "
                f"UTF-8, not ANSI or UTF-16, or restore it from a copy",
            ) from None
        try:
            raw = json.loads(text, parse_constant=_refuse_constant)
        except json.JSONDecodeError as exc:
            raise RunDocumentError(
                path,
                f"is not valid JSON: {exc.msg} (line {exc.lineno}, column "
                f"{exc.colno}); it was left as it is, restore it from a copy",
            ) from None
        except _NonFiniteError as exc:
            # smc writes with allow_nan=False, so a file with one was not
            # written by smc; rewriting it would raise in the middle of a run.
            raise RunDocumentError(
                path, f"contains {exc}, which JSON does not allow"
            ) from None
        if not isinstance(raw, dict):
            raise RunDocumentError(
                path, f"holds a JSON {_json_kind(raw)}, not an object"
            )
        _check_schema_version(path, raw)
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(part) for part in error['loc']) or 'document'}: "
                f"{error['msg']}"
                for error in exc.errors()
            ]
        more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
        raise RunDocumentError(
            path,
            f"{problems[0]}{more}. It was left as it is: upgrade smc if a newer "
            f"one wrote it, or restore it from a copy",
        )

    def provider(self, slot: str) -> StageRecord | None:
        """The record that fills ``slot``: the latest start, whatever its status.

        The latest attempt decides even when it did not finish: a re-run of
        plate registration that failed leaves no usable plate, rather than the
        older one silently. A tie on the start time goes to the later plugin
        name, so the choice never depends on the order of the file.
        """
        filling = [r for r in self.stages.values() if r.provides == slot]
        if not filling:
            return None
        return max(filling, key=lambda r: (r.started_utc, r.plugin))

    def is_stale(self, plugin: str) -> bool:
        """Whether ``plugin`` read an input that has been produced again since.

        True when, for a slot it read, the provider is gone, is another plugin
        or attempt than the one it read, or is itself stale; or when it read
        an optional slot as absent that now has a ``done`` provider. A record
        that read nothing is never stale.
        """
        return self._stale(plugin, frozenset())

    def _stale(self, plugin: str, visited: frozenset[str]) -> bool:
        record = self.stages.get(plugin)
        if record is None or plugin in visited:
            return False
        visited = visited | {plugin}
        for slot, read in record.reads.items():
            current = self.provider(slot)
            if read is None:
                if current is not None and current.status is StageStatus.done:
                    return True
                continue
            if current is None:
                return True
            if (current.plugin, current.attempt) != (read.plugin, read.attempt):
                return True
            if self._stale(current.plugin, visited):
                return True
        return False

    def unusable(self, slot: str, *, profile: str, dry_run: bool) -> str | None:
        """Why ``slot`` cannot be read by a run on ``profile``, with the fix; ``None`` if it can.

        The first row that applies wins: no provider, a provider that did not
        end ``done``, a stale provider, a provider run on another profile, a
        provider that was a dry run when this run is not. A dry run may read
        a real result.
        """
        provider = self.provider(slot)
        if provider is None:
            return (
                f"the run document has no {slot}; smc run --list shows which plugin "
                f"provides it"
            )
        name = provider.plugin
        if provider.status is StageStatus.running:
            return (
                f"the last run of {name} is still running or was interrupted; run "
                f"it again"
            )
        if provider.status is not StageStatus.done:
            return f"the last run of {name} ended {provider.status.value}; run it again"
        if self.is_stale(name):
            return f"{name} read its inputs from an earlier run; run it again"
        if provider.profile != profile:
            return (
                f"{name} was run on {provider.profile}, this is {profile}; run "
                f"{name} again on this profile"
            )
        if provider.dry_run and not dry_run:
            return f"{name} was a dry run; run it without --dry-run"
        return None


def write_stage(path: Path, record: StageRecord) -> RunDocument:
    """Put ``record`` into the document at ``path`` and replace the file atomically.

    The file is read fresh each time (``load_or_new``) and never written from a
    copy held in memory, so the stages another writer added meanwhile
    survive. Two writes at the same instant can still lose one record: one
    writer per document is assumed and not locked (m3 §1.13). The bytes are
    ASCII, so a lone surrogate (FM-05) and ``µ`` are escaped.

    Returns:
        The document that was written.

    Raises:
        RunDocumentError: The document cannot be read (it is left as it is,
            and never replaced) or the file cannot be written. ``record`` of
            the error is the record that was not written.
    """
    try:
        current = RunDocument.load_or_new(path)
        stages = {**current.stages, record.plugin: record}
        written = current.model_copy(update={"stages": stages})
        # `mode="json"` already turns nan and infinity into null (measured), so
        # allow_nan=False is a seatbelt: smc never writes a file `load` refuses.
        text = json.dumps(
            written.model_dump(mode="json"),
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        _write_atomically(path, (text + "\n").encode("ascii"))
    except RunDocumentError as exc:
        raise RunDocumentError(exc.path, exc.reason, exc.kept, record) from exc
    return written


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` by ``data`` so that it is either the old file or the new one (FM-38).

    As ``backends/mm.py::_write_atomically``, which this does not share (the
    five copies are not merged here): a temporary file in the same folder,
    created with ``O_EXCL`` and the mode a plain write gives (``mkstemp``'s
    0600 would survive the rename), then ``os.replace``, all in one ``try`` so
    that a Ctrl-C between two blocks does not leave a file nothing names. On
    a failure the temporary file is removed, except when only the rename
    failed: it then holds the whole new text, so it is kept and the error
    names it. The rename is retried on ``PermissionError`` (FM-24).

    Raises:
        RunDocumentError: The file could not be written; it is as it was.
    """
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.smc-tmp")
    renaming = False
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o666,
        )
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        renaming = True
        _replace(temporary, path)
    except BaseException as exc:
        if renaming and isinstance(exc, OSError):
            raise RunDocumentError(
                path,
                f"could not be replaced ({exc.strerror or exc}); it is as it was, "
                f"and the new text is kept as {temporary}. Close any program that "
                f"holds {path.name} open, then run the stage again",
                kept=temporary,
            ) from exc
        with suppress(OSError):
            temporary.unlink()
        if isinstance(exc, OSError):
            raise RunDocumentError(
                path, f"could not be written ({exc.strerror or exc}); it is as it was"
            ) from exc
        raise


def _replace(source: Path, target: Path) -> None:
    """``os.replace``, tried ``_REPLACE_TRIES`` times while Windows says ``PermissionError``."""
    for attempt in range(1, _REPLACE_TRIES + 1):
        try:
            os.replace(source, target)
        except PermissionError:
            if attempt == _REPLACE_TRIES:
                raise
            time.sleep(_REPLACE_WAIT_S)
        else:
            return


class _NonFiniteError(ValueError):
    """``json.loads`` met ``NaN`` or ``Infinity``, which smc never writes."""


def _refuse_constant(name: str) -> Any:
    raise _NonFiniteError(name)


def _json_kind(value: object) -> str:
    """The JSON name of a decoded value that is not an object."""
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    return "number"


def _check_schema_version(path: Path, raw: dict[str, Any]) -> None:
    """Refuse a ``schema_version`` that is missing, not an integer, below 1 or too new.

    Checked before the model sees the document, so that a newer smc's file is
    reported as newer and not as a list of fields this smc does not know.
    """
    if "schema_version" not in raw:
        raise RunDocumentError(
            path,
            "has no schema_version; smc always writes one, so this file is not a "
            "run document",
        )
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise RunDocumentError(
            path, f"schema_version must be an integer, got {version!r}"
        )
    if version < 1:
        raise RunDocumentError(path, f"schema_version must be 1 or more, got {version}")
    if version > SCHEMA_VERSION:
        raise RunDocumentError(
            path,
            f"written by a newer smc (schema {version}); this smc reads schema "
            f"{SCHEMA_VERSION} — upgrade smc",
        )
