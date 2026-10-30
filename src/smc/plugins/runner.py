"""Running one plugin on one stand and recording what happened (m3 §1.8).

The runner is the path every plugin that moves a stage takes, so its outcomes
are safety-relevant:

* **Everything that needs no stand is checked first** (``check_ready``), and the
  capabilities before anything is written, so a stand that lacks the XY stage
  gets one error naming every missing capability and an untouched run document.
* **The ``running`` record is written before ``run``**: a crash then shows
  ``running``, which ``RunDocument.unusable`` reads as "interrupted", never as a
  result (FM-106).
* **An operator's Ctrl-C is recorded, then re-raised** (FM-71): it is a stop, not
  a result, and the CLI's own mapping to exit 130 must still see it. Any other
  exception is recorded ``failed`` and re-raised as it was, so a ``HardwareError``
  or a ``SafetyRefusedError`` reaches the CLI's exit-code mapping by its type
  (FM-64). Only ``StopRequested``, the plugin's own way out, is an outcome that
  returns.
* **The runner keeps no exception** in a local or an attribute beyond its
  ``except`` block, and the context holds no reference to the microscope: an
  exception kept where its traceback reaches it is a cycle that keeps the core
  alive until the cyclic collector (FM-92, FM-98).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smc import __version__
from smc.hardware.errors import CapabilityMissingError
from smc.plugins.api import (
    Plugin,
    PluginParams,
    Progress,
    RunContext,
    StopRequested,
    _Attempt,
)
from smc.plugins.errors import RequirementsError, ResumeError, UpstreamError
from smc.run.document import (
    ReadRecord,
    RunDocument,
    RunDocumentError,
    StageRecord,
    StageStatus,
    write_stage,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["check_ready", "reachable_checkpoint", "run_plugin"]

log = logging.getLogger("smc.plugins.runner")

_ABSENT = object()


def check_ready(
    plugin: Plugin[Any, Any],
    doc: RunDocument,
    params: PluginParams,
    *,
    profile: str,
    dry_run: bool,
    resume: bool,
) -> None:
    """Refuse a run that cannot start, from the document alone.

    It touches neither hardware nor files, so the CLI calls it before it opens
    the stand (FM-30), and ``run_plugin`` calls it again on the document it
    loads.

    Raises:
        UpstreamError: A slot of ``reads`` is not usable (the first one, in
            the order of ``reads``).
        ResumeError: ``resume`` is true and the plugin has no record, its
            record is ``done``, its parameters or its profile differ, it
            was a dry run and this run is not, or it is stale (what it read
            has been produced again since its checkpoint was made).
    """
    for slot in plugin.reads:
        reason = doc.unusable(slot, profile=profile, dry_run=dry_run)
        if reason is not None:
            raise UpstreamError(plugin.name, slot, reason)
    if not resume:
        return
    name = plugin.name
    record = doc.stages.get(name)
    if record is None:
        raise ResumeError(name, "nothing to resume")
    if record.status is StageStatus.done:
        raise ResumeError(name, f"{name} finished; run it without --resume")
    differing = _differing_keys(record.params, params.model_dump(mode="json"))
    if differing:
        raise ResumeError(
            name,
            f"the parameters differ from the run being resumed: {', '.join(differing)}",
        )
    if record.profile != profile:
        raise ResumeError(
            name,
            f"{name} was run on {record.profile}, this is {profile}; resume it "
            f"there, or run it without --resume",
        )
    if record.dry_run and not dry_run:
        # A dry run's checkpoint stands for work that was never done.
        raise ResumeError(name, f"{name} was a dry run; run it without --resume")
    if doc.is_stale(name):
        # The checkpoint was made against inputs that have been produced again:
        # resuming would finish the work on the new ones and record the whole run
        # as read from them, a false "current" (FM-106).
        raise ResumeError(
            name,
            f"{name} read its inputs from an earlier run, so its checkpoint no "
            f"longer matches them; run it without --resume",
        )


def reachable_checkpoint(
    name: str, doc: RunDocument, *, profile: str, dry_run: bool
) -> StageStatus | None:
    """The status of ``name``'s record when a run afresh would drop a checkpoint ``--resume`` can reach.

    A page that has no ``--resume`` must not start a stopped stage over the work
    the stop saved (#252): an hours-long scan is lost by one click, and no command
    brings it back. It protects only what ``--resume`` could continue, so it uses
    the three tests of ``check_ready`` that need no ``params`` (the profile, a
    dry run's checkpoint for a real run, a stale record); a checkpoint that
    ``--resume`` would refuse anyway is replaced as it always was. The parameters
    are not compared: a stage stopped on purpose is lost by a changed ``step`` as
    surely as by an unchanged one. ``running`` is not covered: it is the run
    itself, or a process that died, and ``unusable`` already says so.

    It touches neither hardware nor files, like ``check_ready``.

    Returns:
        ``StageStatus.stopped`` or ``StageStatus.failed``, or ``None`` when there
        is no such record, no checkpoint, or ``--resume`` would refuse it.
    """
    record = doc.stages.get(name)
    if (
        record is None
        or record.status not in (StageStatus.stopped, StageStatus.failed)
        or record.checkpoint is None
        or record.profile != profile
        or (record.dry_run and not dry_run)
        # The dearest test last: it walks what the record read.
        or doc.is_stale(name)
    ):
        return None
    return record.status


def run_plugin(
    plugin: Plugin[Any, Any],
    microscope: Microscope,
    params: PluginParams,
    run_doc: Path,
    *,
    resume: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
    stop: threading.Event | None = None,
    plugin_version: str | None = None,
) -> StageRecord:
    """Run ``plugin`` on ``microscope`` and record the outcome in ``run_doc``.

    Returns:
        The stage's record: ``done`` with its result, or ``stopped`` when the
        plugin ended with ``StopRequested`` (a stop is an outcome, not an
        error).

    Raises:
        TypeError: ``params`` is not an instance of ``plugin.Params``, or the
            plugin returned something that is not its ``Result``.
        UpstreamError, ResumeError: ``check_ready`` refused the run.
        RequirementsError: The stand lacks capabilities; nothing was written.
        RunDocumentError: The document cannot be read (it is left as it is) or
            written; for the ``done`` write, ``record`` holds the result.
        KeyboardInterrupt: Recorded ``stopped``, then re-raised.
        BaseException: Whatever ``plugin.run`` raised, recorded ``failed``,
            then re-raised unchanged.
    """
    if not isinstance(params, plugin.Params):
        raise TypeError(
            f"{plugin.name} takes {plugin.Params.__name__}, not {type(params).__name__}"
        )
    profile = microscope.profile.microscope.name
    dry_run = microscope.dry_run
    doc = RunDocument.load_or_new(run_doc)
    check_ready(plugin, doc, params, profile=profile, dry_run=dry_run, resume=resume)
    _require(plugin, microscope)

    previous = doc.stages.get(plugin.name)
    checkpoint = previous.checkpoint if resume and previous is not None else None
    if (
        not resume
        and previous is not None
        and previous.checkpoint is not None
        and previous.status in (StageStatus.stopped, StageStatus.failed)
    ):
        log.info(
            "%s: the previous attempt (%s) saved a checkpoint; --resume would "
            "continue it, and this run starts afresh",
            plugin.name,
            previous.status.value,
        )
    running = StageRecord(
        plugin=plugin.name,
        provides=plugin.provides,
        attempt=uuid.uuid4().hex,
        status=StageStatus.running,
        started_utc=datetime.now(timezone.utc),
        profile=profile,
        dry_run=dry_run,
        smc_version=__version__,
        plugin_version=plugin_version,
        params=params.model_dump(mode="json"),
        checkpoint=checkpoint,
        reads=_reads_of(plugin, doc, profile=profile, dry_run=dry_run),
    )
    # Before any hardware is touched: a failure here ends the run.
    write_stage(run_doc, running)

    attempt = _Attempt(running)
    ctx = RunContext(
        plugin=plugin.name,
        run_doc=run_doc,
        dry_run=dry_run,
        snapshot=doc,
        attempt=attempt,
        reads=plugin.reads,
        reads_optional=plugin.reads_optional,
        resumed=checkpoint,
        stop=stop if stop is not None else threading.Event(),
        on_progress=on_progress,
    )
    try:
        result = plugin.run(microscope, params, ctx)
        if not isinstance(result, plugin.Result):
            raise TypeError(
                f"{plugin.name} returned {type(result).__name__}, not "
                f"{plugin.Result.__name__}"
            )
        dumped = result.model_dump(mode="json")
    except StopRequested:
        stopped = _ended(attempt.record, StageStatus.stopped)
        write_stage(run_doc, stopped)
        return stopped
    except KeyboardInterrupt:
        _record_outcome(
            run_doc,
            _ended(
                attempt.record, StageStatus.stopped, error="KeyboardInterrupt: Ctrl-C"
            ),
        )
        raise
    except BaseException as exc:
        # The line, not the error: nothing below may keep it (FM-98).
        line = _error_line(exc)
        _record_outcome(run_doc, _ended(attempt.record, StageStatus.failed, error=line))
        raise
    done = _ended(attempt.record, StageStatus.done, result=dumped)
    # A failure here raises with `record` set to this record: the stage ran, and
    # the caller can still show the result (FM-34).
    write_stage(run_doc, done)
    return done


def _require(plugin: Plugin[Any, Any], microscope: Microscope) -> None:
    """Every capability of ``requires`` that the stand lacks, named at once.

    Nothing has been written when this raises. The detail of each is the
    message ``microscope.require`` gives, which points at ``[roles.assign]``.
    """
    missing = [
        capability for capability in plugin.requires if not microscope.has(capability)
    ]
    if not missing:
        return
    details: list[str] = []
    for capability in missing:
        try:
            microscope.require(capability)
        except CapabilityMissingError as exc:
            details.append(str(exc))
    raise RequirementsError(
        plugin.name, tuple(c.__name__ for c in missing), tuple(details)
    )


def _reads_of(
    plugin: Plugin[Any, Any], doc: RunDocument, *, profile: str, dry_run: bool
) -> dict[str, ReadRecord | None]:
    """Which attempt of which provider each slot is read from, as of now.

    ``check_ready`` found every slot of ``reads`` usable. An optional slot that is
    not usable is recorded as ``None``, which is also what the context answers
    ``upstream_or_none`` with, and logged with its reason.
    """
    reads: dict[str, ReadRecord | None] = {}
    for slot in plugin.reads:
        reads[slot] = _read_of(doc, slot)
    for slot in plugin.reads_optional:
        reason = doc.unusable(slot, profile=profile, dry_run=dry_run)
        if reason is None:
            reads[slot] = _read_of(doc, slot)
        else:
            reads[slot] = None
            log.info("%s: not using %s: %s", plugin.name, slot, reason)
    return reads


def _read_of(doc: RunDocument, slot: str) -> ReadRecord:
    provider = doc.provider(slot)
    if provider is None:  # unreachable once the slot is usable
        raise UpstreamError(
            "the runner",
            slot,
            f"the run document has no {slot}; smc run --list shows which plugin "
            f"provides it",
        )
    return ReadRecord(plugin=provider.plugin, attempt=provider.attempt)


def _ended(
    record: StageRecord,
    status: StageStatus,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> StageRecord:
    """``record`` as it ends: finished now, the checkpoint kept unless the stage is ``done``."""
    return record.model_copy(
        update={
            "status": status,
            "finished_utc": datetime.now(timezone.utc),
            "result": result,
            "error": error,
            "checkpoint": None if status is StageStatus.done else record.checkpoint,
        }
    )


def _record_outcome(run_doc: Path, record: StageRecord) -> None:
    """Write the outcome of a run that ends in an exception; its failure is only logged.

    The exception in flight is what the caller must see: a write that fails
    here must not replace it. The document then still says ``running``, which
    ``unusable`` reads as interrupted.
    """
    try:
        write_stage(run_doc, record)
    except RunDocumentError as exc:
        # Text, never the error: a kept record keeps its traceback (FM-92).
        log.error(
            "%s: the outcome (%s) could not be written to the run document: %s",
            record.plugin,
            record.status.value,
            str(exc),
        )


def _error_line(error: BaseException) -> str:
    """``<Type>: <message>`` on one line, for the record's ``error``."""
    try:
        message = " ".join(str(error).split())
    except Exception:  # a message that cannot be printed is not worth losing the run
        message = ""
    name = type(error).__name__
    return f"{name}: {message}" if message else name


def _differing_keys(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """The parameter keys on which two dumped parameter sets differ, sorted."""
    return sorted(
        key
        for key in old.keys() | new.keys()
        if old.get(key, _ABSENT) != new.get(key, _ABSENT)
    )
