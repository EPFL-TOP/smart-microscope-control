"""``nis-export`` writes the route as an XY position list that NIS-Elements 6.20 loads (m3 §14).

The file's format is ``format.py`` and is pure. This reads the ``route`` record and
moves nothing, so a dry run writes an export like any other. What it decides, because the
file drives the stand once the operator loads it in NIS-Elements and nothing in either
program says that the two agree:

* **Positions are written unconverted.** X, Y and Z are Micro-Manager's readings in µm
  and the offset is the autofocus's own unit. Whether NIS-Elements shares their origin
  and direction is not measured, so there is no sign flip and no offset here, and the
  result says so in ``assumed`` (unknown beats guessed). The owner compares one stop with
  ``smc stage`` at the stand (#198).
* **Z follows what the scan knew.** Every stop with a Z and an offset gives
  ``pfs-offset``; every stop with a Z, ``z``; otherwise ``none``. One flag covers the whole
  list, so a single stop without a Z (a well whose frames were all ``unlocked``, FM-154)
  takes the Z of the others out as well, and the result counts how many stops lacked it.
* **A new file for every run**, as for ``report``: a ``done`` result's file is never
  rewritten (m3 §2.3). Only an ``out`` the operator names is written where they say, and
  refused when it exists unless ``overwrite`` is true. Its folder is never created, so a
  mistyped one is refused. It is recorded by its absolute path, which may name the
  person's account (FM-50); that is the operator's own choice here.
* Nothing is checkpointed: a stop ends it before the write and leaves no file, and a
  write that fails says what is kept and to run ``nis-export`` again.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.nis_export.format import SUFFIX, NisPosition, encode
from smc.plugins.slots import NisExportResult, RoutePosition, RouteResult
from smc.run.files import (
    AtomicWriteError,
    new_token,
    relative_to_run,
    write_atomically,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from smc.hardware.microscope import Microscope

__all__ = ["NisExportParams", "NisExportPlugin"]

_NAME = "nis-export"

#: What the export rests on that was not measured; read at the stand (#198).
_FRAME = (
    "x_um, y_um and z_um are Micro-Manager's readings in µm and pfs_offset is "
    "PFSOffset's own unit, written unconverted: that NIS-Elements shares their "
    "origin and direction is not measured"
)
_BOTH_FLAGS = (
    "bIncludeZ and bPFSEnabled are both set, which the NIS-Elements sample did not show"
)


class NisExportParams(PluginParams):
    """Where the list goes, and whether it may replace a file that is there."""

    out: Path | None = Field(
        None,
        description=(
            "The file to write, ending in .xml (a folder NIS-Elements reads); "
            "default nis-export/<token>.xml in the run folder."
        ),
    )
    overwrite: bool = Field(False, description="Replace out when it exists.")

    @field_validator("out")
    @classmethod
    def _ends_in_xml(cls, value: Path | None) -> Path | None:
        """Refuse an ``out`` that is not an ``.xml`` file, ``out=`` (the current folder) included."""
        if value is not None and value.suffix.lower() != SUFFIX:
            raise ValueError(f"out must end with {SUFFIX}, not {value.name!r}")
        return value


class NisExportPlugin:
    """Writes the route's positions, in order, as a multipoint list for NIS-Elements."""

    name = _NAME
    summary = "Write the route as an XY position list (multipoint) that NIS-Elements 6.20 loads."
    Params = NisExportParams
    Result = NisExportResult
    requires: tuple[type, ...] = ()
    provides = "nis-export"
    reads: tuple[str, ...] = ("route",)
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: NisExportParams,
        ctx: RunContext,
    ) -> NisExportResult:
        """Write the list and name its file: ``nis-export/<token>.xml``, or ``out``.

        Raises:
            PluginRunError: The route has no positions or is not numbered ``0..n-1``,
                a position cannot be written (the message says to run ``route``
                again), ``out`` cannot be resolved, is a folder, or exists without
                ``overwrite``, or the file cannot be written (the message says
                what is kept).
            StopRequested: The operator stopped it, before the write.
        """
        route = ctx.upstream("route", RouteResult)
        stops = route.positions
        _require_numbered(stops)
        mode = _z_mode(stops)
        positions = [_nis_position(stop, mode) for stop in stops]

        problem = ""
        data = b""
        try:
            data = encode(positions)
        except ValueError as exc:
            problem = f"{exc}; run route again"
        if problem:
            # Raised outside the handler: nothing here keeps the encoder's error (FM-98).
            raise PluginRunError(_NAME, problem)

        target = _target(params)
        ctx.raise_if_stopped()
        file = _write(ctx, data, target)
        ctx.logger.info(
            "NIS-Elements export written: %s (%d positions, z_mode %s)",
            file,
            len(positions),
            mode,
        )
        return NisExportResult(
            file=file,
            positions=len(positions),
            z_mode=mode,
            assumed=_assumed(stops, mode),
        )


def _require_numbered(stops: Sequence[RoutePosition]) -> None:
    """Refuse a route with no stop, or whose stops are not numbered ``0..n-1`` in order.

    The file is written in the list's order, which is the route's order: a record edited
    or replaced behind the runner (FM-111) would be exported in an order its ``order``
    fields contradict.
    """
    if not stops:
        raise PluginRunError(
            _NAME,
            "the route has no positions, so there is nothing to export; run route "
            "again with wells that hold cells of the wanted labels",
        )
    for index, stop in enumerate(stops):
        if stop.order != index:
            raise PluginRunError(
                _NAME,
                f"the route's position {index} has order {stop.order}; run route again",
            )


def _z_mode(stops: Sequence[RoutePosition]) -> Literal["z", "pfs-offset", "none"]:
    """What the export carries for Z: both values when every stop has both, else Z, else nothing."""
    if all(stop.z_um is not None for stop in stops):
        if all(stop.pfs_offset is not None for stop in stops):
            return "pfs-offset"
        return "z"
    return "none"


def _nis_position(stop: RoutePosition, mode: str) -> NisPosition:
    return NisPosition(
        name=stop.name,
        x_um=stop.x_um,
        y_um=stop.y_um,
        z_um=stop.z_um if mode != "none" else None,
        pfs_offset=stop.pfs_offset if mode == "pfs-offset" else None,
    )


def _assumed(stops: Sequence[RoutePosition], mode: str) -> tuple[str, ...]:
    """The frame of the coordinates, then what the chosen mode takes from the stops."""
    count = len(stops)
    if mode == "pfs-offset":
        return (_FRAME, _BOTH_FLAGS)
    if mode == "z":
        lacking = sum(stop.pfs_offset is None for stop in stops)
        return (
            _FRAME,
            f"{lacking} of {count} positions have no pfs_offset: the export carries Z "
            f"without the autofocus (bPFSEnabled false, dPFSOffset written as 0)",
        )
    lacking = sum(stop.z_um is None for stop in stops)
    return (
        _FRAME,
        f"{lacking} of {count} positions have no z_um: the export carries neither Z "
        f"nor an autofocus offset (bIncludeZ and bPFSEnabled false, both written as 0); "
        f"NIS-Elements keeps the focus it has",
    )


def _target(params: NisExportParams) -> Path | None:
    """The file ``out`` names, resolved from the current folder, or ``None`` for the default.

    A folder is refused with or without ``overwrite``, and an existing file without it. An
    ``OSError`` from the filesystem is the same refusal as one from ``resolve``: Windows
    re-raises a few network errors out of both (a share that drops, FM-116).
    """
    if params.out is None:
        return None
    problem = ""
    target = params.out
    try:
        target = params.out.resolve()
        is_folder = target.is_dir()
        exists = target.exists()
    except OSError as exc:
        problem = (
            f"cannot resolve out {params.out} ({exc.strerror or exc}); check the path, "
            f"then run nis-export again"
        )
    else:
        if is_folder:
            problem = (
                f"{target.as_posix()} is a folder; out names the file to write, "
                f"ending in {SUFFIX}"
            )
        elif exists and not params.overwrite:
            problem = (
                f"{target.as_posix()} exists; pass overwrite=true to replace it, or "
                f"choose another out"
            )
    if problem:
        # Raised outside the handler: nothing here keeps the filesystem's error (FM-98).
        raise PluginRunError(_NAME, problem)
    return target


def _shown(ctx: RunContext, path: Path) -> str:
    """``path`` relative to the run folder when it is inside it, else its absolute POSIX path."""
    try:
        return relative_to_run(ctx.run_dir, path)
    except (ValueError, OSError):
        return path.as_posix()


def _write(ctx: RunContext, data: bytes, target: Path | None) -> str:
    """Write the list and return the name the result records.

    The write is all or nothing (FM-38). Only the rename failing, which on Windows is a
    program holding the file (NIS-Elements may), leaves the whole new list in a temporary
    file that the message names. The default file is under a new token and its folder is
    made; an ``out`` is where the operator chose and its folder must be there.
    """
    if target is None:
        path = ctx.run_dir / "nis-export" / f"{new_token()}{SUFFIX}"
        file = relative_to_run(ctx.run_dir, path)
        make_parents = True
    else:
        path = target
        file = target.as_posix()
        make_parents = False
    where = _shown(ctx, path)
    problem = ""
    try:
        write_atomically(path, data, make_parents=make_parents)
    except AtomicWriteError as exc:
        reason = exc.strerror or exc
        if exc.kept is not None:
            problem = (
                f"cannot write {where} ({reason}); the whole export is kept as "
                f"{_shown(ctx, exc.kept)}; close any program that holds {where} "
                f"(NIS-Elements may), then run nis-export again"
            )
        else:
            problem = (
                f"cannot write {where} ({reason}); check that its folder exists and "
                f"can be written, then run nis-export again"
            )
    if problem:
        # Raised outside the handler: nothing here keeps the write's error (FM-98).
        raise PluginRunError(_NAME, problem)
    return file
