"""What a plugin is, and what it gets besides the microscope (m3 §1.3, §1.4).

A plugin is a class with a name, a one-line summary, two pydantic models, the
capabilities it requires, the slot of the run document it fills and the slots it
reads, and one method, ``run(microscope, params, ctx)``: the hardware loop of
ADR-0004. The algorithm it applies is a pure function in another module that
takes no ``Microscope``.

The context is how a plugin talks to whoever runs it, in four ways that never
reach for the hardware or the operator's terminal directly: progress goes to a
callback, a stop is asked of an ``Event``, what upstream stages produced is read
from a snapshot of the run document, and a checkpoint is written through the
run document's own atomic write.

Nothing here imports ``smc.hardware`` at run time (``Microscope`` is only a type)
and nothing imports ``pymmcore_plus``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from smc.plugins.errors import ResumeError, UpstreamError
from smc.run.document import RunDocument, StageRecord, write_stage

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "NAME_PATTERN",
    "M",
    "P",
    "Plugin",
    "PluginParams",
    "PluginResult",
    "Progress",
    "R",
    "RunContext",
    "StopRequested",
]

#: A plugin's name, its slot and a slot it reads: kebab-case, matched in full.
#: The name is the entry point's name, keys the plugin's record in the run
#: document and names its logger.
NAME_PATTERN = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"

#: ``wait_s`` waits in slices of at most this long: on Windows one long lock
#: wait is not interrupted by Ctrl-C (CPython's documented behaviour, FM-83), so
#: the loop returns to Python between two slices.
_WAIT_SLICE_S = 0.1


class PluginParams(BaseModel):
    """Base of every plugin's parameters.

    ``allow_inf_nan=False`` makes every float finite even where the plugin
    forgot a bound (FM-32); unknown keys are refused so that a misspelt
    ``--param`` is an error and not a default; the model is frozen so a run's
    parameters are the ones recorded in the run document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PluginResult(BaseModel):
    """Base of every plugin's result.

    Unknown is ``None``, never ``nan`` ("unknown beats guessed"). ``assumed``
    names what the result rests on that was taken as given and not measured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    #: What the result rests on that was assumed rather than measured.
    assumed: tuple[str, ...] = ()


P = TypeVar("P", bound=PluginParams)
R = TypeVar("R", bound=PluginResult)
M = TypeVar("M", bound=BaseModel)


@runtime_checkable
class Plugin(Protocol[P, R]):
    """The interchangeable tool (ADR-0004); the registry checks it by ``isinstance``.

    ``requires`` are capability Protocols (keys of ``CAPABILITY_ROLES``);
    ``provides`` is the slot of the run document the result fills, by
    convention the plugin's name unless several plugins produce the same
    thing; ``reads`` are slots the runner insists on and ``reads_optional``
    slots the plugin uses when they are usable. ``run`` returns its ``Result``
    and never writes the run document itself.
    """

    name: str
    summary: str
    Params: type[P]
    Result: type[R]
    requires: tuple[type, ...]
    provides: str
    reads: tuple[str, ...]
    reads_optional: tuple[str, ...]

    def run(self, microscope: Microscope, params: P, ctx: RunContext) -> R:
        """The hardware loop (ADR-0004): returns the result, never writes the document."""
        ...


@dataclass(frozen=True, slots=True)
class Progress:
    """One report of a plugin's progress; ``total`` is ``None`` when not known."""

    plugin: str
    done: int
    total: int | None
    message: str


class StopRequested(Exception):  # noqa: N818 (a signal, like StopIteration; the name is fixed by m3 §1.3)
    """A plugin's way out when ``ctx.should_stop()`` is true.

    It derives from ``Exception`` and from nothing in ``smc.plugins.errors`` or
    ``smc.hardware.errors``, so an ``except PluginError`` or ``except
    SafetyRefusedError`` in a loop never swallows a stop (FM-64).
    """


class _Attempt:
    """The running record as it is now, shared by the runner and its context.

    ``save_checkpoint`` replaces the record, and the runner needs the last
    checkpoint when the attempt ends ``stopped`` or ``failed``: the checkpoint
    is the work not to repeat.
    """

    __slots__ = ("record",)

    def __init__(self, record: StageRecord) -> None:
        self.record = record


class RunContext:
    """What a plugin gets besides the microscope; built by the runner, never by a plugin.

    The constructor is the runner's and keyword-only, not public API: a plugin
    is tested through ``run_plugin`` with a run document under ``tmp_path``.
    The context holds the stop event, the progress callback, a snapshot of the
    run document and the run document's path, and no reference to the
    microscope (FM-92).

    Attributes:
        plugin: The plugin's name.
        logger: ``smc.plugins.<name>``. The context adds no handler and sets
            no level (FM-46): the CLI's ``-v`` shows it.
        run_dir: The folder of the run document. A plugin that writes files
            writes them under it and stores their paths relative to it, so the
            folder moves as a whole.
        dry_run: ``microscope.dry_run``, for a plugin that must label what it
            did not do.
    """

    plugin: str
    logger: logging.Logger
    run_dir: Path
    dry_run: bool

    def __init__(
        self,
        *,
        plugin: str,
        run_doc: Path,
        dry_run: bool,
        snapshot: RunDocument,
        attempt: _Attempt,
        reads: tuple[str, ...],
        reads_optional: tuple[str, ...],
        resumed: dict[str, Any] | None,
        stop: threading.Event,
        on_progress: Callable[[Progress], None] | None,
    ) -> None:
        self.plugin = plugin
        self.logger = logging.getLogger(f"smc.plugins.{plugin}")
        self.run_dir = run_doc.parent
        self.dry_run = dry_run
        self._run_doc = run_doc
        self._snapshot = snapshot
        self._attempt = attempt
        self._reads = reads
        self._reads_optional = reads_optional
        # Which optional slots the runner found usable at the start, as the
        # running record says: ``None`` is a slot that was not.
        self._read_at_start = dict(attempt.record.reads)
        self._resumed = resumed
        self._stop = stop
        self._on_progress = on_progress

    def progress(self, done: int, total: int | None = None, message: str = "") -> None:
        """Report progress to the runner's callback; never validated, never fatal.

        A cosmetic bug must not end a scan, so a callback that raises an
        ``Exception`` is logged once at WARNING and not called again in this
        run; ``KeyboardInterrupt`` passes.
        """
        callback = self._on_progress
        if callback is None:
            return
        try:
            callback(Progress(self.plugin, done, total, message))
        except Exception as exc:
            self._on_progress = None
            # Text, never the error: a kept record keeps its traceback (FM-92).
            self.logger.warning(
                "the progress callback failed (%s: %s); progress is not reported "
                "any more in this run",
                type(exc).__name__,
                exc,
            )

    def should_stop(self) -> bool:
        """Whether a stop was asked for (a UI's Stop, a test); the CLI has none.

        On the command line Ctrl-C is the ``KeyboardInterrupt`` that the
        ``Executor`` already turns into a stop of any move under way (M1 §9).
        """
        return self._stop.is_set()

    def raise_if_stopped(self) -> None:
        """Raise ``StopRequested`` when ``should_stop()`` is true."""
        if self._stop.is_set():
            raise StopRequested

    def wait_s(self, seconds: float) -> None:
        """Wait up to ``seconds``, in slices of at most 0.1 s, ending at a stop.

        A plugin never calls ``time.sleep``: it could not be stopped, and on
        Windows one long wait keeps Ctrl-C from being delivered (FM-83).

        Raises:
            ValueError: ``seconds`` is not finite or is negative.
            StopRequested: The stop event is set, before or during the wait.
        """
        if not (math.isfinite(seconds) and seconds >= 0):
            raise ValueError(
                f"wait_s needs a finite number of seconds, 0 or more, got {seconds!r}"
            )
        deadline = time.monotonic() + seconds
        while True:
            self.raise_if_stopped()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._stop.wait(min(_WAIT_SLICE_S, remaining))

    def upstream(self, slot: str, model: type[M]) -> M:
        """The result of the provider of ``slot``, validated by ``model``.

        Read from the snapshot of the run document that the runner took
        before ``run``, after it checked every slot of ``reads`` was usable.

        Raises:
            ValueError: ``slot`` is not in the plugin's ``reads``: a plugin
                bug, since the declaration is what the runner checked.
            UpstreamError: ``model`` refuses the result; run the provider
                again.
        """
        if slot not in self._reads:
            raise ValueError(
                f"{self.plugin} does not declare {slot!r} in reads, so the runner "
                f"did not check it; add it to reads (or to reads_optional, and "
                f"call upstream_or_none)"
            )
        return self._result_of(slot, model)

    def upstream_or_none(self, slot: str, model: type[M]) -> M | None:
        """``upstream`` for a slot of ``reads_optional``; ``None`` when it was not usable.

        Raises:
            ValueError: ``slot`` is not in the plugin's ``reads_optional``.
            UpstreamError: The slot was usable and ``model`` refuses its result.
        """
        if slot not in self._reads_optional:
            raise ValueError(
                f"{self.plugin} does not declare {slot!r} in reads_optional; add "
                f"it there (or to reads, and call upstream)"
            )
        if self._read_at_start.get(slot) is None:
            return None
        return self._result_of(slot, model)

    def _result_of(self, slot: str, model: type[M]) -> M:
        provider = self._snapshot.provider(slot)
        if provider is None or provider.result is None:
            raise UpstreamError(
                self.plugin,
                slot,
                f"the run document has no {slot}; smc run --list shows which "
                f"plugin provides it",
            )
        try:
            return model.model_validate(provider.result)
        except ValidationError as exc:
            problem = _first_problem(exc)
        # Raised outside the handler: nothing here keeps the validation error.
        raise UpstreamError(
            self.plugin,
            slot,
            f"the result of {provider.plugin} is not what {self.plugin} expects "
            f"({problem}); run {provider.plugin} again",
        )

    def save_checkpoint(self, state: BaseModel) -> None:
        """Record ``state`` as the running stage's checkpoint, atomically.

        Call it after each unit of work the plugin would not want to repeat (a
        well, a field). It goes through ``write_stage``, which reads the file
        again, so a checkpoint never reverts a stage written meanwhile.

        Raises:
            RunDocumentError: The document cannot be written; the run ends.
        """
        record = self._attempt.record.model_copy(
            update={"checkpoint": state.model_dump(mode="json")}
        )
        write_stage(self._run_doc, record)
        self._attempt.record = record

    def checkpoint(self, model: type[M]) -> M | None:
        """The checkpoint of the attempt being resumed, validated by ``model``.

        ``None`` when the run was not started with ``resume=True`` or the
        previous attempt saved none. It stays what it was at the start of this
        attempt, whatever ``save_checkpoint`` writes meanwhile.

        Raises:
            ResumeError: ``model`` refuses the saved checkpoint.
        """
        if self._resumed is None:
            return None
        try:
            return model.model_validate(self._resumed)
        except ValidationError as exc:
            problem = _first_problem(exc)
        raise ResumeError(
            self.plugin,
            f"the saved checkpoint is not what {self.plugin} expects ({problem}); "
            f"run it again without --resume",
        )


def _first_problem(error: ValidationError) -> str:
    """``<where>: <what>`` of the first validation error, on one line."""
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "result"
    return f"{where}: {first['msg']}"
