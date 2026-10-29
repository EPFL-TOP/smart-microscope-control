"""``smc`` — the command line is the reference interface of this project.

Every plugin and every hardware capability is reachable from here before it
is reachable from any graphical interface (ADR-0006), because a command line
is scriptable, testable and works over SSH to a microscope PC.
"""

from __future__ import annotations

import codecs
import contextlib
import errno
import gc
import importlib
import json
import logging
import math
import os
import platform
import secrets
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Literal, TextIO

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from smc import __version__
from smc.hardware import core as core_mod
from smc.hardware.backends import nikon
from smc.hardware.backends.mm import as_hardware_errors
from smc.hardware.capabilities import XY, Autofocus, Camera, Shutter, XYStage, ZStage
from smc.hardware.errors import (
    CoreError,
    HardwareError,
    ProfileError,
    SafetyRefusedError,
    StopNote,
    StopOutcome,
    join_names,
    stop_notes,
)
from smc.hardware.microscope import Microscope, MicroscopeState
from smc.hardware.profile import (
    DEMO_NAME,
    ROLES_WITH_A_STOP,
    Profile,
    list_profiles,
    search_paths,
)
from smc.hardware.roles import Role
from smc.plugins import (
    ParamsError,
    PluginError,
    PluginNotFoundError,
    Progress,
    ResumeError,
    UpstreamError,
    check_ready,
    describe_params,
    parse_params,
    run_plugin,
)
from smc.plugins import discover as discover_plugins
from smc.plugins import load as load_plugin
from smc.run import RunDocument, RunDocumentError, StageRecord, StageStatus

app = typer.Typer(
    name="smc",
    help="Control several microscopes through one layer; run interchangeable tools.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    # A --debug traceback must not dump every frame's locals, the core included.
    pretty_exceptions_show_locals=False,
)
console = Console()

#: Click settings for the commands that take signed numbers: ``-100`` is a
#: distance, not an unknown option. A misspelt option then fails as a bad
#: number or an extra argument, which is still a usage error (exit 2).
_NUMBERS: dict[str, Any] = {"ignore_unknown_options": True}

ProfileOption = Annotated[
    str,
    typer.Option(
        "--profile",
        "-p",
        envvar="SMC_PROFILE",
        help="The profile to open: a name found on the search paths (see smc "
        "profiles) or the path to a .toml file.",
    ),
]
DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Log the move and do not send it: the stage stays where it is.",
    ),
]

if TYPE_CHECKING:
    from smc.discovery.config_build import ConfigPlan

    _StreamHandler = logging.StreamHandler[TextIO]
else:  # StreamHandler is subscriptable only from Python 3.11
    _StreamHandler = logging.StreamHandler


class _CliLogHandler(_StreamHandler):
    """The handler the CLI puts on the ``smc`` logger; marked so it can be replaced.

    Each command builds a new one on ``sys.stderr`` as it is at that moment:
    ``CliRunner`` swaps the stream for every invocation, and a handler bound
    to an earlier one writes into a closed buffer.
    """


@dataclass(frozen=True)
class _Settings:
    """The root options, handed to every command through ``ctx.obj``."""

    debug: bool = False


def _configure_logging(level: int) -> Callable[[], None]:
    """Send the ``smc`` loggers to stderr at ``level``; return the call that undoes it.

    The root logger is left alone, and records still propagate, so an
    application or ``pytest`` that configured it keeps receiving them. The
    undo runs when the command ends: ``app`` also runs inside other programs
    (``CliRunner``), where a handler left on a stream that was closed turns
    every later record into a "Logging error" traceback.
    """
    logger = logging.getLogger("smc")
    for handler in list(logger.handlers):
        if isinstance(handler, _CliLogHandler):
            logger.removeHandler(handler)
    previous_level = logger.level
    handler = _CliLogHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)

    def undo() -> None:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    return undo


# --- output ------------------------------------------------------------------
# Nothing below goes through rich markup (FM-36): exception messages, device
# labels and role lines carry brackets ("[roles.assign]", "[core]") that rich
# would silently drop. Nothing is wrapped either (FM-37): a one-line error
# must stay one line in a redirected log.


def _say(text: str) -> None:
    """Print a line verbatim: no markup, no highlighting, no emoji codes, no wrapping."""
    console.print(text, markup=False, highlight=False, emoji=False, soft_wrap=True)


def _fail(message: str, *, indent: str = "") -> None:
    """Print ``✗ message``; only the mark is styled."""
    console.print(Text.assemble(indent, ("✗ ", "red"), message), soft_wrap=True)


def _ok(message: str) -> None:
    """Print ``✓ message``; only the mark is styled."""
    console.print(Text.assemble(("✓ ", "green"), message), soft_wrap=True)


def _aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    """Columns padded to their widest cell, two spaces apart, no trailing blanks."""
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    ]


def _um(value: float) -> str:
    """Two decimals; a stage that reads ``-0.0`` at the origin shows ``0.00``."""
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def _xy(position: XY) -> str:
    return f"XY ({_um(position.x_um)}, {_um(position.y_um)}) µm"


def _z(z_um: float) -> str:
    return f"Z {_um(z_um)} µm"


# --- errors ------------------------------------------------------------------

#: Exit codes (design §9): 1 for a hardware, profile or file error, 2 for a
#: refusal (nothing was sent; Click's usage errors exit 2 as well), 130 for
#: Ctrl-C, as a shell reports a process ended by SIGINT.
_EXIT_ERROR = 1
_EXIT_REFUSED = 2
_EXIT_INTERRUPTED = 130


class _Reported:
    """Turn what a command body raises into one ``✗`` line and an exit code.

    Wrapped around the ``with Microscope.open(...)`` block, so the stand is
    released before the line is printed. Ctrl-C is the ``KeyboardInterrupt``
    on which the Executor sent a stop to any move under way (FM-17) and noted
    what that stop did, which the line says (m2 §11.6); there is no signal
    handler (FM-70).

    A class, not a ``@contextmanager`` generator: a generator that converts an
    exception into another keeps the command's frames, and the stand with
    them, in a reference cycle on Python 3.11+ (through contextlib's frame),
    which only the cyclic collector frees, at the exit of the process (#148,
    FM-92). It keeps no exception on ``self`` for the same reason.

    Args:
        ctx: The command's context; ``--debug`` lets every error through,
            so Typer prints the traceback.
        force_hint: What the operator types to force a forceable refusal; it
            replaces the API's ``how_to_force`` (``force=True``).
        after_report: Called with the error once its ``✗`` line is printed,
            for a command that has more to say after it (``doctor``'s
            diagnosis of a stand that would not open). Not called under
            ``--debug``, which prints the traceback instead of the line.
    """

    def __init__(
        self,
        ctx: typer.Context,
        *,
        force_hint: str = "",
        after_report: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._ctx = ctx
        self._force_hint = force_hint
        self._after_report = after_report

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        # Never swallows: it lets the exception through or raises ``typer.Exit``.
        if exc is None or isinstance(
            exc, typer.Exit | typer.Abort | typer.BadParameter
        ):
            return False
        if not isinstance(exc, Exception | KeyboardInterrupt):
            return False
        settings = self._ctx.find_object(_Settings)
        if settings is not None and settings.debug:
            return False
        code = _report(exc, self._force_hint)
        if self._after_report is not None:
            self._after_report(exc)
        raise typer.Exit(code=code) from None


def _report(exc: BaseException, force_hint: str) -> int:
    """Print the one line for ``exc``; return the exit code."""
    if isinstance(exc, SafetyRefusedError):
        fix = f" — {force_hint or exc.how_to_force}" if exc.how_to_force else ""
        _fail(f"{exc.reason}{fix}")
        return _EXIT_REFUSED
    if isinstance(exc, HardwareError):
        _fail(str(exc))
        return _EXIT_ERROR
    if isinstance(exc, KeyboardInterrupt):
        _fail(_interrupted_line(stop_notes(exc)))
        return _EXIT_INTERRUPTED
    # Plugin errors (m3 §1.9): a refusal the operator fixes with a command exits 2,
    # a plugin that does not load, a stand that lacks a capability and a run
    # document that cannot be read or written exit 1.
    if isinstance(exc, PluginNotFoundError | ParamsError | UpstreamError | ResumeError):
        _fail(str(exc))
        return _EXIT_REFUSED
    if isinstance(exc, PluginError | RunDocumentError):
        _fail(str(exc))
        return _EXIT_ERROR
    # The plate page's server that ended alone (m3 §15.3). Looked up, not imported:
    # `smc` starts without the ui modules, and the error exists only once they are.
    controller = sys.modules.get("smc.ui.controller")
    if controller is not None and isinstance(exc, controller.PageServerError):
        _fail(str(exc))
        return _EXIT_ERROR
    _fail(
        f"unexpected {type(exc).__name__}: {exc} — run it again as smc --debug … "
        "for the traceback, and file an issue"
    )
    return _EXIT_ERROR


def _interrupted_line(notes: Sequence[StopNote]) -> str:
    """The Ctrl-C line: what each stop the interrupt caused did (m2 §11.6, #206).

    It said "any move under way was stopped" whatever the stop did, and on
    the Ti2-E, which refuses every stop, the stage went the full 20000 µm
    (#208, FM-117). The layer sends a move only inside the ``Executor``'s
    ``_move``, and every Ctrl-C there leaves a note, so no note means that no
    move was under way.
    """
    read = "read the position (smc stage get, smc z get) before the next move"
    if not notes:
        return f"interrupted — no move was under way; {read}"
    clauses = "; ".join(_note_clause(note) for note in notes)
    if any(note.outcome is not StopOutcome.accepted for note in notes):
        read = f"let it finish or stop it with the stand's own controls, then {read}"
    return f"interrupted — {clauses}; {read}"


def _note_clause(note: StopNote) -> str:
    """One stop of the Ctrl-C line, in the words of m2 §11.6."""
    name = note.motion
    if note.outcome is StopOutcome.accepted:
        return f"{name} was stopped"
    if note.outcome is StopOutcome.refused:
        if note.declared:
            return (
                f"{name} refuses a stop on this stand ([safety] stop_refused_by), "
                f"so it runs to its target"
            )
        return f"{name} refused the stop, so it may still be moving"
    if note.outcome is StopOutcome.impossible:
        return f"{name} cannot be stopped from smc, so it may still be moving"
    return f"the stop of {name} was cut by a second Ctrl-C, so it may still be moving"


def _stop_refused_line(roles: Iterable[Role]) -> str | None:
    """The ``!`` line before a move on a role whose stop the stand refuses; ``None`` if none.

    Printed once the stand is open and before anything is sent, never in
    dry-run (m2 §11.6): on such a stand, Ctrl-C ends smc and not the move,
    and the operator must know it before the move, not after it.
    """
    declared = set(roles)
    named = [role.value for role in ROLES_WITH_A_STOP if role in declared]
    if not named:
        return None
    return (
        f"! the profile declares that this stand refuses stop() on "
        f"{join_names(named)} ([safety] stop_refused_by): once sent, a move runs "
        f"to its target, and Ctrl-C ends smc, not the move"
    )


def _say_stop_refused(
    microscope: Microscope, roles: Iterable[Role] | None = None
) -> None:
    """Print ``_stop_refused_line`` for ``roles`` the profile declares (every one by default)."""
    declared = microscope.profile.safety.stop_refused_by
    line = _stop_refused_line(
        declared if roles is None else [r for r in roles if r in declared]
    )
    if line is not None:
        _say(line)


def tolerate_unencodable_output(stream: TextIO) -> None:
    """Let a non-UTF-8 stream replace what it cannot encode instead of crashing.

    A redirected stream on Windows gets the ANSI code page (cp1252) with
    strict errors, and ``✓`` / ``✗`` have no cp1252 encoding: ``smc doctor >
    f.txt`` used to die with ``UnicodeEncodeError`` (#42). Only the error
    handler changes; the encoding is the one the operator's shell chose.
    """
    try:
        encoding = codecs.lookup(stream.encoding or "").name
    except LookupError:
        encoding = ""
    reconfigure = getattr(stream, "reconfigure", None)
    if encoding != "utf-8" and reconfigure is not None:
        reconfigure(errors="replace")


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Log on stderr, one line per command sent to a device.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Log everything, and show the traceback instead of the "
            "one-line error.",
        ),
    ] = False,
) -> None:
    """Control several microscopes through one layer; run interchangeable tools."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            tolerate_unencodable_output(stream)
    ctx.call_on_close(
        _configure_logging(
            logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
        )
    )
    ctx.obj = _Settings(debug=debug)


@app.command()
def version() -> None:
    """Print the versions that matter when reporting a problem."""
    # A version number, not hardware access: the one sanctioned import here.
    import pymmcore_plus  # noqa: TID251

    table = Table(show_header=False, box=None)
    table.add_row("smc", __version__)
    table.add_row("pymmcore-plus", pymmcore_plus.__version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())
    console.print(table)


@app.command()
def profiles(ctx: typer.Context) -> None:
    """List the profiles found on the search paths, and the search paths.

    Reads files only; no stand is opened. A profile that does not load is
    listed with its error, so its owner sees what to fix; the command still
    succeeds. A TOML file without a microscope table is not a profile and is
    skipped without a word (./pyproject.toml is on the search path).
    """
    with _Reported(ctx):
        found: list[tuple[str, Path | None]] = list(list_profiles())
        if all(name != DEMO_NAME for name, _ in found):
            found = sorted([*found, (DEMO_NAME, None)], key=lambda entry: entry[0])
        lines = _aligned(
            [[name, str(path) if path else "(built in)"] for name, path in found]
        )
        for line, (_, path) in zip(lines, found, strict=True):
            _say(line)
            if path is not None and (error := _load_error(path)):
                _fail(error, indent="  ")
        _say("Search paths:")
        for directory in search_paths():
            _say(f"  {directory}" + ("" if directory.is_dir() else " (not found)"))


def _load_error(path: Path) -> str:
    """Why the profile at ``path`` does not load, on one line; empty when it loads."""
    try:
        Profile.load(path)
    except ProfileError as exc:
        message = str(exc).removeprefix(f"{path}: ")
    else:
        return ""
    return " ".join(line.strip() for line in message.splitlines() if line.strip())


@app.command()
def devices(ctx: typer.Context, profile: ProfileOption = DEMO_NAME) -> None:
    """List the loaded devices and the roles they fill, then the role table.

    The table shows, for each role, the device chosen, the other candidates
    and where the choice came from, so a wrong pick can be fixed in the
    profile.
    """
    with _Reported(ctx), Microscope.open(profile) as microscope:
        filled: dict[str, list[str]] = {}
        for role in Role:
            label = microscope.roles.get(role)
            if label is not None:
                filled.setdefault(label, []).append(role.value)
        rows = [["label", "type", "library/name", "roles"]]
        rows += [
            [
                device.label,
                device.type,
                f"{device.library}/{device.name}",
                ", ".join(filled.get(device.label, [])),
            ]
            for device in microscope.devices
        ]
        for line in _aligned(rows):
            _say(line)
        _print_roles(microscope)


def _print_roles(microscope: Microscope) -> None:
    """The role table, then each role warning.

    The warnings are logged as well; printing them here keeps them in a
    report whose stderr was not redirected with it.
    """
    _say("Roles")
    for line in microscope.roles.describe():
        _say(f"  {line}")
    for warning in microscope.roles.warnings:
        _say(f"! {warning}")


def _status_line(microscope: Microscope, state: MicroscopeState) -> str:
    """``Microscope.describe()``'s line, built from a snapshot already taken.

    describe() reads the stand again, and doctor must print the snapshot
    whose errors decide its exit code. The format is describe()'s, kept
    equal by a test; a failed read shows ``?``, its reason is in the errors.
    """
    parts: list[str] = []
    if microscope.has(XYStage):
        xy = state.xy
        parts.append("XY ?" if xy is None else f"XY ({xy.x_um:.1f}, {xy.y_um:.1f}) µm")
    if microscope.has(ZStage):
        parts.append("Z ?" if state.z_um is None else f"Z {state.z_um:.2f} µm")
    if microscope.has(Camera):
        exposure_ms = state.exposure_ms
        parts.append("exposure ?" if exposure_ms is None else f"{exposure_ms:g} ms")
    if microscope.has(Shutter):
        if state.shutter_open is None:
            parts.append("shutter ?")
        else:
            parts.append("shutter open" if state.shutter_open else "shutter closed")
    if microscope.has(Autofocus):
        engaged, locked = state.autofocus_engaged, state.autofocus_locked
        if engaged is None or (engaged and locked is None):
            parts.append("autofocus ?")
        elif not engaged:
            parts.append("autofocus off")
        else:
            parts.append("autofocus locked" if locked else "autofocus on, not locked")
    if state.moving:
        parts.append("moving: " + ", ".join(state.moving))
    if state.halted:
        parts.append("HALTED")
    return " | ".join(parts) or "no stage, camera or shutter"


def _install_dirs(status: core_mod.MicroManagerStatus) -> list[Path]:
    """The install's adapter folder, for the end of the list MMCore searches."""
    return [status.install_dir] if status.install_dir is not None else []


def _nikon_checks(
    config: Path | None, adapter_dirs: Sequence[Path]
) -> list[nikon.StandCheck]:
    """The file checks of each Nikon stand that ``config`` loads (design m2 §7.4)."""
    if config is None:
        return []
    return [
        nikon.check_stand(stand, adapter_dirs)
        for stand in nikon.stands_in_config(config)
    ]


def _print_nikon_readiness(checks: Sequence[nikon.StandCheck]) -> None:
    """The adapter and driver rows of each Nikon stand, before it is opened.

    They are information: a stand that opens is a stand that opens, whatever
    the file checks said, so they never change the exit code.
    """
    for check in checks:
        _say(f"{check.stand.name} ({check.stand.adapter})")
        for line in _aligned([[name, text] for name, text in nikon.readiness(check)]):
            _say(f"  {line}")


def _diagnose_nikon(checks: Sequence[nikon.StandCheck], error: BaseException) -> None:
    """What to check when a Nikon stand's configuration would not open.

    Only for a ``CoreError``, which is what a failed open raises: any other
    error is reported as it always was. It runs once the open has failed and
    MMCore has unloaded what it had loaded, so no probe competes with a core
    for the stand. The diagnosis is help: if it fails itself, that is one more
    line, never the loss of the error it follows.
    """
    if not isinstance(error, CoreError):
        return
    try:
        for check in checks:
            probes = nikon.deep_check(check, extra_path=check.searched)
            _say(f"{check.stand.name}: what to check")
            for line in nikon.diagnosis(check, probes):
                _say(f"  {line}")
    except KeyboardInterrupt:
        # The probe killed its child before it re-raised. Said here, since the
        # report of the body's own Ctrl-C is already behind us.
        _fail("interrupted while checking the stand; the remaining checks were not run")
        raise typer.Exit(code=_EXIT_INTERRUPTED) from None
    except Exception as exc:
        _fail(f"could not diagnose the stand: {exc}")


@app.command()
def doctor(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="A Micro-Manager .cfg to test-load as it is, without a profile. "
            "It wins over --profile.",
        ),
    ] = None,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Check that this machine can drive a microscope, or at least the simulator.

    Without --config, it opens the profile and prints its role table and the
    stand's state. Exits 1 when Micro-Manager is missing, the stand does not
    open or a device does not answer a read, so it can gate a session script.
    A role warning is printed and still exits 0.

    A configuration that loads a Nikon stand (from the profile, or --config)
    gets its adapter and driver checks printed before it is opened; they never
    change the exit code. When it does not open, what to check follows, with
    Windows' own reason when a DLL will not load.
    """
    st = core_mod.status()
    table = Table(title="Micro-Manager", show_header=False, box=None)
    table.add_row(
        "install", str(st.install_dir) if st.install_dir else "[red]NOT FOUND[/red]"
    )
    table.add_row("device API", st.api_version)
    table.add_row("core", st.core_version)
    table.add_row("adapters", str(len(st.adapters)))
    table.add_row(
        "demo devices",
        "[green]yes[/green]" if st.has_demo_devices else "[yellow]no[/yellow]",
    )
    console.print(table)

    if not st.installed:
        _fail(core_mod.INSTALL_HINT)
        raise typer.Exit(code=1)

    if config is None:
        profile_checks: list[nikon.StandCheck] = []
        with _Reported(
            ctx, after_report=lambda error: _diagnose_nikon(profile_checks, error)
        ):
            loaded = Profile.load(profile)
            # The profile's folders first, then the install: MMCore's order (FM-76).
            profile_checks += _nikon_checks(
                loaded.config_path(),
                [*loaded.adapter_search_dirs(), *_install_dirs(st)],
            )
            _print_nikon_readiness(profile_checks)
            with Microscope.open(loaded) as microscope:
                name = loaded.microscope.name
                _say(f"Profile {name} ({loaded.source or 'built in'})")
                _print_roles(microscope)
                if loaded.sample is not None:
                    # Inside the branch: only this stand loads the module (m1 §8).
                    from smc.testing.sample_stand import sample_lines

                    for line in sample_lines(loaded.sample):
                        _say(line)
                _say_stop_refused(microscope)
                # One read decides the exit code and is the line printed: a
                # flaky device read twice could show "XY ?" beside the ✓.
                state = microscope.state()
                _say(_status_line(microscope, state))
                errors = state.errors
        if errors:
            for error in errors:
                _fail(error)
            raise typer.Exit(code=_EXIT_ERROR)
        _ok(f"{name} loads and answers.")
        return

    target = str(config)
    # With no profile, the install alone is searched for the adapter.
    config_checks = _nikon_checks(config, _install_dirs(st))
    _print_nikon_readiness(config_checks)
    # No profile, so no role: the raw core is read through the same proxy
    # under the name "core" (#120). _Reported wraps the whole block, so the
    # core is released before the one line is printed, and a bug that is not
    # a device fault is reported like on every other command.
    with (
        _Reported(
            ctx, after_report=lambda error: _diagnose_nikon(config_checks, error)
        ),
        core_mod.opened(config) as core,
    ):
        reads = as_hardware_errors(core, "core")
        roles = {
            "camera": reads.getCameraDevice(),
            "xy stage": reads.getXYStageDevice(),
            "focus": reads.getFocusDevice(),
            "autofocus": reads.getAutoFocusDevice(),
            "shutter": reads.getShutterDevice(),
        }
        rt = Table(title=f"Core roles — {target}", show_header=False, box=None)
        for role, device in roles.items():
            rt.add_row(role, device or "[yellow]— not set —[/yellow]")
        # A fault on the position read is a finding, not the end of the
        # inventory: the table still says which device is the XY stage, hence
        # which cable. One read decides the table and the verdict (FM-39).
        fault: HardwareError | None = None
        if roles["xy stage"]:
            try:
                x, y = reads.getXYPosition()
            except HardwareError as exc:
                fault = exc
            else:
                rt.add_row("xy position", f"({x:.1f}, {y:.1f}) µm")
        console.print(rt)
        try:
            if fault is not None:
                raise fault
        finally:
            # Exception -> traceback -> this frame -> this local: a cycle that
            # keeps the core until the cyclic collector (#148, FM-92).
            del fault
    _ok(f"{target} loads and answers.")


@app.command("install-driver")
def install_driver(
    ctx: typer.Context,
    stand: Annotated[str, typer.Argument(help="The stand: ti2 or tie.")],
    source: Annotated[
        Path | None,
        typer.Option(
            "--from",
            help="The driver DLL to copy; default: the one in the vendor's SDK folder.",
        ),
    ] = None,
    to: Annotated[
        Path | None,
        typer.Option(
            "--to",
            help="The folder of the stand's adapter; default: its folder in the "
            "Micro-Manager install.",
        ),
    ] = None,
) -> None:
    """Copy a Nikon stand's driver DLL to the folder its adapter loads it from.

    The Ti2 adapter loads Ti2_Mic_Driver.dll from its own folder, and the copy
    in the Ti2 SDK folder is not enough. This copies, never moves, and never
    overwrites a DLL that differs. The Ti-E adapter finds its driver on the
    system path, so there is nothing to copy for it. Exits 0 when the driver is
    in place, 1 when it is not.
    """
    with _Reported(ctx):
        found = nikon.stand_for_key(stand)
        if found is None:
            keys = ", ".join(known.key for known in nikon.STANDS)
            _fail(f"unknown stand '{stand}'; use one of: {keys}")
            raise typer.Exit(code=_EXIT_REFUSED)
        destination = to
        if destination is None and found.driver.beside_adapter:
            install = core_mod.find_install()
            if install is None:
                _fail(
                    f"there is no Micro-Manager install to take the folder of "
                    f"{found.adapter_file} from; pass --to with that folder"
                )
                raise typer.Exit(code=_EXIT_ERROR)
            if not (install / found.adapter_file).is_file():
                _fail(
                    f"{found.adapter_file} is not in the Micro-Manager install "
                    f"({install}); pass --to with the folder that holds it"
                )
                raise typer.Exit(code=_EXIT_ERROR)
            destination = install
        # A stand whose driver is on the system path is refused before the
        # folder is looked at, so any folder will do for it.
        ok, message = nikon.install_driver(
            found, destination if destination is not None else Path(), source=source
        )
        if not ok:
            _fail(message)
            raise typer.Exit(code=_EXIT_ERROR)
        _ok(message)


@app.command()
def discover(
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Folder to (over)write inventory.json and inventory.txt in, "
            "UTF-8. Named after the stand, e.g. local/surveys/nikon-ti2.",
        ),
    ] = None,
    all_adapters: Annotated[
        bool,
        typer.Option(
            "--all-adapters",
            help="List the devices of every installed adapter (slow: loads "
            "every DLL). Default: the lab's adapters and the hinted ones.",
        ),
    ] = False,
    probe_adapter: Annotated[
        str | None,
        typer.Option(
            "--probe-adapter",
            help="Load and initialise every device of ONE adapter. Contacts "
            "the hardware: stand powered, vendor software closed.",
        ),
    ] = None,
    port: Annotated[
        str | None,
        typer.Option(
            "--port",
            help="Serial port to give, before they initialise, to the probed "
            "devices that need one (COM3, /dev/ttyUSB0). Needs --probe-adapter.",
        ),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option(
            "--timeout-s",
            min=1.0,
            max=3600.0,
            help="How long the adapter child process may run (1-3600 s).",
        ),
    ] = 180.0,
    no_os: Annotated[
        bool, typer.Option("--no-os", help="Skip serial, USB / PnP and PCI.")
    ] = False,
) -> None:
    """Survey this PC: OS devices, vendor hints, Micro-Manager adapters.

    Read-only: nothing is initialised unless --probe-adapter is given, and
    nothing is written unless --out is. A failing tool or adapter becomes a
    note in the report, never the end of the survey.
    """
    from smc.discovery import inventory
    from smc.discovery import report as report_mod
    from smc.discovery.mm_inventory import check_port

    if not math.isfinite(timeout_s):  # click's range check lets nan through
        _fail(f"--timeout-s must be a number of seconds, not {timeout_s}")
        raise typer.Exit(code=2)
    if port is not None:
        if not probe_adapter:
            _fail("--port needs --probe-adapter")
            raise typer.Exit(code=2)
        try:
            port = check_port(port)
        except ValueError as exc:
            _fail(str(exc))
            raise typer.Exit(code=2) from None
    # --out is checked before the survey and written before the report is
    # printed: a broken pipe or Ctrl-C while printing must not lose the files.
    if out is not None:
        try:
            report_mod.prepare(out)
        except OSError as exc:
            _fail(f"cannot write to {out}: {exc}")
            raise typer.Exit(code=1) from None
    if probe_adapter:
        # Text, not markup: the adapter and the port are what the owner typed.
        where = f" (port {port})" if port is not None else ""
        console.print(
            Text.assemble(
                ("! ", "yellow"),
                f"Probing {probe_adapter}{where}: the stand must be powered and "
                "its vendor software closed.",
            ),
            soft_wrap=True,
        )
    with console.status("Surveying this PC…"):
        inv = inventory(
            all_adapters=all_adapters,
            probe_adapter=probe_adapter,
            probe_port=port,
            include_os=not no_os,
            timeout_s=timeout_s,
        )
    written: tuple[Path, Path] | None = None
    write_error: OSError | None = None
    if out is not None:
        try:
            written = report_mod.write(inv, out)
        except OSError as exc:
            # The survey took minutes: a failed write costs the files, not
            # the report on screen.
            write_error = exc
    for item in report_mod.renderables(inv):
        console.print(item)
    if write_error is not None:
        _fail(f"could not write the survey to {out}: {write_error}")
        # Dropped with its traceback's frame once the line is printed (#148, FM-92).
        del write_error
        raise typer.Exit(code=1)
    if written is not None:
        json_path, text_path = written
        _ok(f"wrote {json_path} and {text_path.name}")


# --- config build ---------------------------------------------------------------
# `smc config build` initialises every device of one adapter, in the adapter child
# of `smc discover`, and writes a Micro-Manager configuration from the ones that
# answer (design m2 §10). Nothing in it moves a stage.

config_app = typer.Typer(
    help="Micro-Manager configuration files.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(config_app, name="config")

#: ``BUILD_TIMEOUT_S`` of ``smc.discovery.mm_inventory``, which is not imported at
#: module level (it loads pydantic models); a test keeps the two equal.
_BUILD_TIMEOUT_DEFAULT_S = 300.0


def _adapter_name(value: str, option: str) -> str:
    """An adapter named on the command line, or a usage error (exit 2)."""
    from smc.discovery.mm_inventory import check_device_name

    try:
        return check_device_name(value)
    except ValueError as exc:
        said = str(exc).replace("the device name", "the adapter name")
        _fail(f"{option}: {said}")
        raise typer.Exit(code=_EXIT_REFUSED) from None


def _stand_adapter(loaded: Profile) -> str:
    """The adapter of the one Nikon stand the profile's configuration loads.

    Without ``--adapter`` a build needs to know which adapter's devices to
    initialise; guessing among several would contact a stand nobody named.
    """
    config = loaded.config_path()
    if config is None:
        why = f"profile {loaded.microscope.name} loads the demo configuration"
    else:
        stands = nikon.stands_in_config(config)
        if len(stands) == 1:
            return stands[0].adapter
        why = (
            f"{config} loads no Nikon stand"
            if not stands
            else f"{config} loads {len(stands)} Nikon stands: "
            + ", ".join(stand.adapter for stand in stands)
        )
    _fail(
        f"name the stand's adapter with --adapter ({why}; smc discover lists the "
        "installed ones)"
    )
    raise typer.Exit(code=_EXIT_REFUSED)


def _backup_path(out: Path) -> Path:
    return out.with_name(out.name + ".bak")


def _check_build_out(out: Path, *, replace: bool, writing: bool) -> None:
    """Refuse a destination that cannot be written, before anything is initialised.

    The build is minutes of hardware work and the file is its whole result
    (FM-30). An existing file is refused unless it is to be replaced
    (``--force``) or only read (``--dry-run``).
    """
    if out.is_dir():
        _fail(f"{out} is a folder; name the file to write")
        raise typer.Exit(code=_EXIT_REFUSED)
    if not out.parent.is_dir():
        _fail(f"the folder {out.parent} does not exist")
        raise typer.Exit(code=_EXIT_REFUSED)
    if out.exists() and not replace:
        _fail(
            f"{out} exists: pass --force to replace it (its presets, pixel sizes "
            f"and labels are carried over, and it is saved as {_backup_path(out)})"
        )
        raise typer.Exit(code=_EXIT_REFUSED)
    if writing:
        try:
            with tempfile.TemporaryFile(dir=out.parent):
                pass
        except OSError as exc:
            _fail(f"cannot write to {out.parent}: {exc}")
            raise typer.Exit(code=_EXIT_ERROR) from None


def _write_config(out: Path, text: str, *, keep_previous: bool) -> Path | None:
    """Replace ``out`` by ``text``, after saving what was there as ``<out>.bak``.

    The ``.bak`` comes first, so that the previous bytes are saved before they
    are replaced, and both go through a temporary file and ``os.replace``
    (FM-38). When the ``.bak`` cannot be made (Windows antivirus or an editor
    holds it, FM-24), ``out`` is not replaced, and the new text, the result of
    minutes of hardware work, is kept beside it as ``<out>.new``: it must be on
    disk somewhere that the message names (FM-34). The text is UTF-8 with
    ``surrogateescape`` (a carried cp1252 byte comes back as it was) and LF line
    ends on every OS: MMCore loads either (measured).

    Returns:
        The ``.bak`` path when the previous file was saved.

    Raises:
        OSError: ``out`` could not be written; the message says what was kept.
    """

    def write(partial: Path) -> None:
        partial.write_text(
            text, encoding="utf-8", errors="surrogateescape", newline="\n"
        )

    backup = None
    if keep_previous:
        backup = _backup_path(out)
        try:
            _save_backup(out, backup)
        except OSError as exc:
            kept = out.with_name(out.name + ".new")
            why = f"could not save the previous file as {backup}: {exc.strerror or exc}"
            try:
                _replace(kept, write)
            except OSError as second:
                raise OSError(
                    f"{why}; and the new configuration could not be kept as "
                    f"{kept} either: {second.strerror or second}"
                ) from exc
            raise OSError(
                f"{why}; {out} was not replaced, and the new configuration is "
                f"kept as {kept}"
            ) from exc
    _replace(out, write)
    return backup


def _save_backup(out: Path, backup: Path) -> None:
    """Copy ``out`` onto ``backup`` through a temporary file, or leave ``backup`` as it was.

    ``_replace`` would keep the temporary file when the final rename fails and
    word it as "the new file": here that file holds the old bytes, and the
    message would lie about what it is.
    """
    partial = backup.with_name(f".{backup.name}.{secrets.token_hex(4)}.part")
    try:
        shutil.copyfile(out, partial)
        os.replace(partial, backup)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


def _printable(text: str) -> str:
    """``text`` as a UTF-8 console can print it.

    A line carried from the previous file may hold bytes that are not UTF-8,
    read as lone surrogates (``surrogateescape``), which a strict UTF-8 stream
    refuses; the file keeps the bytes, the screen shows a replacement character.
    """
    return text.encode("utf-8", errors="surrogateescape").decode(
        "utf-8", errors="replace"
    )


def _carried_summary(plan: ConfigPlan, out: Path) -> str | None:
    """``carried over from <out>: 3 presets, 2 pixel sizes, labels of 4 devices``."""
    counts = {"preset": 0, "pixel": 0, "labels": 0, "device": 0}
    for block in plan.carried:
        counts[block.kind] += 1

    def many(n: int, one: str, more: str) -> str:
        return f"{n} {one if n == 1 else more}"

    parts = []
    if counts["preset"]:
        parts.append(many(counts["preset"], "preset", "presets"))
    if counts["pixel"]:
        parts.append(many(counts["pixel"], "pixel size", "pixel sizes"))
    if counts["labels"]:
        parts.append(many(counts["labels"], "device's labels", "devices' labels"))
    if counts["device"]:
        parts.append(
            many(
                counts["device"],
                "delay or focus direction",
                "delays or focus directions",
            )
        )
    if not parts:
        return None
    return f"carried over from {out}: " + ", ".join(parts)


@config_app.command("build")
def config_build(
    ctx: typer.Context,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="The configuration file to write, in a folder that exists. An "
            "existing file needs --force: its presets, pixel sizes and labels are "
            "carried over, and it is saved as <out>.bak.",
        ),
    ],
    profile: ProfileOption = DEMO_NAME,
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help="The stand's adapter, whose devices are all initialised. "
            "Default: the one Nikon stand the profile's configuration loads.",
        ),
    ] = None,
    camera_adapter: Annotated[
        str | None,
        typer.Option(
            "--camera-adapter",
            help="The adapter of a camera to add (smc discover lists the "
            "installed ones). Only its cameras, and their hub, are written.",
        ),
    ] = None,
    skip: Annotated[
        list[str] | None,
        typer.Option(
            "--skip",
            help="A device never to load, as the walk names it; repeat it. For "
            "a device that crashes the build.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the configuration and write nothing. An existing --out "
            "is read, to show what is carried over.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Replace an existing --out, saved first as <out>.bak."
        ),
    ] = False,
    timeout_s: Annotated[
        float,
        typer.Option(
            "--timeout-s",
            min=1.0,
            max=3600.0,
            help="How long the build child process may run (1-3600 s).",
        ),
    ] = _BUILD_TIMEOUT_DEFAULT_S,
) -> None:
    """Write a Micro-Manager configuration from the devices that answer.

    Initialises every device of the stand's adapter (and the cameras of
    --camera-adapter), so the stand must be powered and its vendor software
    closed. Use --dry-run first. A device that crashes the build is named, and
    --skip leaves it out on the next run. A hub that will not initialise writes
    nothing and prints the likely causes. Nothing here moves a stage.
    """
    from smc.discovery import report as report_mod
    from smc.discovery.config_build import (
        cut_short,
        plan_config,
        read_previous,
        render_config,
    )
    from smc.discovery.mm_inventory import build_section, check_device_name

    def say(text: str) -> None:
        _say(_printable(text))

    # Usage first, then the destination: both before any child starts.
    if not math.isfinite(timeout_s):  # click's range check lets nan through
        _fail(f"--timeout-s must be a number of seconds, not {timeout_s}")
        raise typer.Exit(code=_EXIT_REFUSED)
    skips: list[str] = []
    for name in skip or []:
        try:
            skips.append(check_device_name(name))
        except ValueError as exc:
            _fail(f"--skip: {exc}")
            raise typer.Exit(code=_EXIT_REFUSED) from None
    named_camera = (
        _adapter_name(camera_adapter, "--camera-adapter")
        if camera_adapter is not None
        else None
    )
    named_stand = _adapter_name(adapter, "--adapter") if adapter is not None else None
    with _Reported(ctx):
        loaded = Profile.load(profile)
        stand = named_stand if named_stand is not None else _stand_adapter(loaded)
        search_dirs = loaded.adapter_search_dirs()
    if named_camera is not None and named_camera == stand:
        _fail(
            f"--camera-adapter is the stand's adapter ({stand}): its devices "
            "are initialised once; name the camera's own adapter"
        )
        raise typer.Exit(code=_EXIT_REFUSED)
    _check_build_out(out, replace=force or dry_run, writing=not dry_run)
    try:
        previous = read_previous(out)
    except OSError as exc:
        _fail(f"cannot read {out}: {exc}")
        raise typer.Exit(code=_EXIT_ERROR) from None

    where = stand + (f" and {named_camera}" if named_camera else "")
    console.print(
        Text.assemble(
            ("! ", "yellow"),
            f"Building from {where}: every device is initialised, so the "
            "stand must be powered and its vendor software closed.",
        ),
        soft_wrap=True,
    )
    try:
        with console.status("Building…"):
            run = build_section(
                stand,
                camera=named_camera,
                skip=skips,
                search_paths=search_dirs,
                timeout_s=timeout_s,
            )
    except KeyboardInterrupt:
        # build_section killed the child's whole tree before it re-raised.
        _fail("interrupted — the build child was stopped; nothing was written")
        raise typer.Exit(code=_EXIT_INTERRUPTED) from None

    scan = run.scan
    plan = (
        plan_config(
            scan,
            previous=previous,
            overrides=loaded.roles.assign,
            exclusions=loaded.roles.exclude,
            skip=skips,
        )
        if scan is not None
        else None
    )
    text: str | None = None
    if plan is not None and plan.writable:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        text = render_config(
            plan,
            header=[
                f"Generated by smc config build on {stamp} from {where}",
                f"Profile {loaded.microscope.name}",
            ],
        )

    # The file comes before the report: a broken pipe or a Ctrl-C while the
    # report is printed must not lose minutes of hardware work (FM-30).
    backup: Path | None = None
    write_error: OSError | None = None
    if text is not None and not dry_run:
        try:
            backup = _write_config(out, text, keep_previous=previous is not None)
        except OSError as exc:
            write_error = exc

    if scan is not None:
        for probe in (scan.stand, scan.camera):
            if probe is not None:
                for item in report_mod.probe_renderables(probe):
                    console.print(item)
    notes = [*(scan.notes if scan else []), *(plan.notes if plan else [])]
    for note in [*notes, *run.notes]:
        say(f"- {note}")
    if plan is not None and plan.writable:
        say("Roles")
        for line in plan.roles.describe():
            say(f"  {line}")
        for warning in plan.roles.warnings:
            say(f"! {warning}")
        summary = _carried_summary(plan, out)
        if summary is not None:
            say(summary)
        for block, why in plan.not_carried:
            say(f"! set aside: {block.key}: {why}")

    if write_error is not None:
        _fail(f"could not write {out}: {write_error}")
        # Dropped with its traceback's frame once the line is printed (#148, FM-92).
        del write_error
        raise typer.Exit(code=_EXIT_ERROR)
    if text is None:
        refusals = (
            plan.refusals if plan is not None else ["the build child left no result"]
        )
        for refusal in refusals:
            _fail(_printable(refusal))
        if run.stopped_at and (scan is None or cut_short(scan)):
            say(f"re-run with --skip {run.stopped_at} to leave it out")
            if any("did not finish in" in note for note in run.notes):
                # The deadline stopped it on whatever it was doing: a slow device
                # is not a crashing one, and leaving it out costs the file.
                say(
                    f"or, if {run.stopped_at} was only slow, raise --timeout-s "
                    f"(it was {timeout_s:g} s)"
                )
        raise typer.Exit(code=_EXIT_ERROR)
    if dry_run:
        say(f"--- would write {out} ---")
        for line in text.splitlines():
            say(line)
        return
    saved = f", previous saved as {backup}" if backup is not None else ""
    _ok(f"wrote {out}{saved}")
    say(f"check it with: smc doctor --config {out}")


# --- stage and z ---------------------------------------------------------------
# Every move opens the stand, moves, and releases it: positions are read back
# from the device, never remembered between two commands. No short option
# but -p: with ignore_unknown_options, Click reads "-inf" letter by letter,
# so a registered letter would turn part of a number into an option.

stage_app = typer.Typer(
    help="The XY stage, in µm (X right, Y up).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
z_app = typer.Typer(
    help="The focus drive, in µm (Z up).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(stage_app, name="stage")
app.add_typer(z_app, name="z")

#: For a jog the fix is the CLI flag, not the API's ``force=True``.
_JOG_FORCE_HINT = "re-run with --force if the distance is intended"


def _move_xy(
    ctx: typer.Context,
    profile: str,
    dry_run: bool,
    move: Callable[[XYStage], XY],
    force_hint: str = "",
) -> None:
    """Open the stand, run ``move`` on its XY stage, print where it landed."""
    with (
        _Reported(ctx, force_hint=force_hint),
        Microscope.open(profile, dry_run=dry_run) as microscope,
    ):
        stage = microscope.require(XYStage)
        if not dry_run:
            _say_stop_refused(microscope, [Role.xy_stage])
        landed = move(stage)
        # In dry-run the move returns the commanded target; the stage itself
        # is read again to show that it did not move.
        here = stage.position_um() if dry_run else None
    if here is None:
        _say(_xy(landed))
    else:
        _say(f"[dry-run] would move to {_xy(landed)}; the stage is at {_xy(here)}")


def _move_z(
    ctx: typer.Context,
    profile: str,
    dry_run: bool,
    move: Callable[[ZStage], float],
    force_hint: str = "",
) -> None:
    """Open the stand, run ``move`` on its focus drive, print where it landed."""
    with (
        _Reported(ctx, force_hint=force_hint),
        Microscope.open(profile, dry_run=dry_run) as microscope,
    ):
        drive = microscope.require(ZStage)
        if not dry_run:
            _say_stop_refused(microscope, [Role.focus])
        landed = move(drive)
        here = drive.position_um() if dry_run else None
    if here is None:
        _say(_z(landed))
    else:
        _say(f"[dry-run] would move to {_z(landed)}; the drive is at {_z(here)}")


@stage_app.command("get")
def stage_get(ctx: typer.Context, profile: ProfileOption = DEMO_NAME) -> None:
    """Print where the XY stage reports it is, in µm."""
    with _Reported(ctx), Microscope.open(profile) as microscope:
        position = microscope.require(XYStage).position_um()
    _say(_xy(position))


@stage_app.command("move", context_settings=_NUMBERS)
def stage_move(
    ctx: typer.Context,
    x_um: Annotated[float, typer.Argument(help="The target X, in µm.")],
    y_um: Annotated[float, typer.Argument(help="The target Y, in µm.")],
    dry_run: DryRunOption = False,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Move the XY stage to a position in µm; the profile's soft limits apply."""
    _move_xy(ctx, profile, dry_run, lambda stage: stage.move_to_um(x_um, y_um))


@stage_app.command("jog", context_settings=_NUMBERS)
def stage_jog(
    ctx: typer.Context,
    dx_um: Annotated[
        float, typer.Argument(help="How far to move in X, in µm (negative: left).")
    ],
    dy_um: Annotated[
        float, typer.Argument(help="How far to move in Y, in µm (negative: down).")
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Move further than the profile's safety.max_jog_um allows.",
        ),
    ] = False,
    dry_run: DryRunOption = False,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Move the XY stage by a distance in µm; a long jog needs --force.

    A jog longer than the profile's safety.max_jog_um is refused, because a
    typo in a relative move (1000 for 100) is how a stage leaves the well.
    """
    _move_xy(
        ctx,
        profile,
        dry_run,
        lambda stage: stage.move_by_um(dx_um, dy_um, force=force),
        force_hint=_JOG_FORCE_HINT,
    )


@z_app.command("get")
def z_get(ctx: typer.Context, profile: ProfileOption = DEMO_NAME) -> None:
    """Print where the focus drive reports it is, in µm."""
    with _Reported(ctx), Microscope.open(profile) as microscope:
        z_um = microscope.require(ZStage).position_um()
    _say(_z(z_um))


@z_app.command("move", context_settings=_NUMBERS)
def z_move(
    ctx: typer.Context,
    z_um: Annotated[float, typer.Argument(help="The target Z, in µm.")],
    dry_run: DryRunOption = False,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Move the focus drive to a position in µm; the profile's soft limits apply."""
    _move_z(ctx, profile, dry_run, lambda drive: drive.move_to_um(z_um))


@z_app.command("jog", context_settings=_NUMBERS)
def z_jog(
    ctx: typer.Context,
    dz_um: Annotated[
        float, typer.Argument(help="How far to move, in µm (negative: down).")
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Move further than the profile's safety.max_z_jog_um allows.",
        ),
    ] = False,
    dry_run: DryRunOption = False,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Move the focus drive by a distance in µm; a long jog needs --force.

    A jog longer than the profile's safety.max_z_jog_um is refused, because a
    typo in a focus jog (1000 for 10) drives the objective into the sample.
    --force passes that guard only: the profile's safety.z_soft_limits_um
    still bound the target.
    """
    _move_z(
        ctx,
        profile,
        dry_run,
        lambda drive: drive.move_by_um(dz_um, force=force),
        force_hint=_JOG_FORCE_HINT,
    )


# --- snap ----------------------------------------------------------------------

#: A snap cannot be interrupted (FM-32): ``snapImage`` holds the driver for
#: the whole exposure, so the CLI refuses one longer than a minute.
_MAX_SNAP_EXPOSURE_MS = 60_000.0


def _check_out(path: Path) -> Path:
    """Refuse a destination that cannot be written, before the stand is opened (FM-30)."""
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise typer.BadParameter(f"{path} must end in .tif or .tiff")
    if path.is_dir():
        raise typer.BadParameter(f"{path} is a folder; name the file to write")
    if not path.parent.is_dir():
        raise typer.BadParameter(f"the folder {path.parent} does not exist")
    return path


def _replace(out: Path, write: Callable[[Path], object]) -> None:
    """Have ``write`` fill a new file next to ``out``, then move it onto ``out``.

    tifffile opens its target with ``"wb"``, which truncates it at once: a
    Ctrl-C or an error half-way through would leave a broken file where an
    earlier capture was. The new file gets the mode a plain write gives
    (0o666 less the umask); ``mkstemp``'s 0600 would survive the rename and
    leave every capture readable by its owner only. When the rename fails
    (Windows refuses to replace a TIFF that Fiji holds open), ``out`` stays
    as it was and the new file is kept: it holds a frame that was taken.
    On any other failure, a Ctrl-C before the rename included, the new file
    is removed, since no message would name it.

    Raises:
        OSError: The folder cannot be written, or ``out`` cannot be replaced;
            the message then names the file that was kept.
    """
    partial = out.with_name(f".{out.name}.{secrets.token_hex(4)}.part")
    # O_EXCL: never write into a file that something else created.
    os.close(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    written = False
    try:
        write(partial)
        written = True
        os.replace(partial, out)
    except BaseException as exc:
        if written and isinstance(exc, OSError):
            # strerror alone: str(exc) already names both files.
            raise OSError(
                f"{exc.strerror or exc}; the new file is kept as {partial}"
            ) from exc
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


def _check_exposure(value_ms: float | None) -> float | None:
    """Refuse an exposure that is not finite or not in (0, 60 000] ms (FM-32)."""
    if value_ms is not None and not (
        math.isfinite(value_ms) and 0 < value_ms <= _MAX_SNAP_EXPOSURE_MS
    ):
        raise typer.BadParameter(
            f"must be more than 0 and at most {_MAX_SNAP_EXPOSURE_MS:g} ms, "
            f"got {value_ms:g}"
        )
    return value_ms


@app.command()
def snap(
    ctx: typer.Context,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            callback=_check_out,
            help="The TIFF to write, .tif or .tiff, in a folder that exists. "
            "An existing file is overwritten.",
        ),
    ] = Path("frame.tif"),
    exposure_ms: Annotated[
        float | None,
        typer.Option(
            "--exposure-ms",
            callback=_check_exposure,
            help="The exposure to set first, in ms (more than 0, at most "
            "60000). Default: the camera's current exposure.",
        ),
    ] = None,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Snap one frame and write it as a TIFF, with its exposure in ms.

    The frame keeps the camera's own dtype (uint8 or uint16) and is never
    rescaled. The TIFF's metadata holds the profile, the camera, the
    exposure, the pixel size in µm (0.0 when unknown) and the stage position.
    """
    write_error: OSError | None = None
    with _Reported(ctx):
        import tifffile

        with Microscope.open(profile) as microscope:
            camera = microscope.require(Camera)
            if exposure_ms is not None:
                camera.set_exposure_ms(exposure_ms)
            frame_exposure_ms = camera.exposure_ms()
            frame = camera.snap()
            if not (
                frame.ndim == 2
                and frame.dtype.kind == "u"
                and frame.dtype.itemsize <= 2
            ):
                raise HardwareError(
                    f"the camera returned a {frame.dtype} frame of shape "
                    f"{frame.shape}; smc snap writes 2-D uint8 or uint16 frames"
                )
            state = microscope.state()
            meta: dict[str, object] = {
                "smc_version": __version__,
                "profile": microscope.profile.microscope.name,
                "camera": microscope.roles.get(Role.camera),
                "exposure_ms": float(frame_exposure_ms),
                # 0.0 is the camera saying "unknown"; it is written as such.
                "pixel_size_um": state.pixel_size_um,
                "xy_um": None if state.xy is None else [state.xy.x_um, state.xy.y_um],
                "z_um": state.z_um,
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            height, width = frame.shape
            # Everything printed later is computed here: nobody measured
            # whether the frame's buffer survives the devices being unloaded.
            summary = (
                f"{height}x{width} {frame.dtype}, exposure {frame_exposure_ms:g} ms, "
                f"min {int(frame.min())}, max {int(frame.max())}"
            )
            try:
                _replace(out, lambda path: tifffile.imwrite(path, frame, metadata=meta))
            except OSError as exc:
                # FM-34: the frame was taken; the summary still gets printed.
                write_error = exc
    if write_error is None:
        _say(f"wrote {out}: {summary}")
        return
    _say(f"snapped (not written): {summary}")
    _fail(f"could not write {out}: {write_error}")
    # Dropped with its traceback's frame once the line is printed (#148, FM-92).
    del write_error
    raise typer.Exit(code=_EXIT_ERROR)


# --- run -----------------------------------------------------------------------
# `smc run` runs one plugin on the stand and records the outcome in a run document
# (design m3 §1.10). Everything that needs no stand is checked before the stand is
# opened (FM-30), and the result is printed once it is released.

#: The progress printer's pace: the first call, then one a second, and always the last.
_PROGRESS_INTERVAL_S = 1.0


def _progress_printer() -> Callable[[Progress], None]:
    """A progress callback that prints plain text, at most a line a second.

    The first call is printed, then a call at least a second after the last
    printed one, and every call that reaches its total: a fast loop prints its
    first and last lines, a long one a line a second, and a scan of a plate does
    not drown the result.
    """
    last_printed: float | None = None

    def show(progress: Progress) -> None:
        nonlocal last_printed
        now = time.monotonic()
        finished = progress.total is not None and progress.done == progress.total
        if (
            last_printed is not None
            and not finished
            and now - last_printed < _PROGRESS_INTERVAL_S
        ):
            return
        last_printed = now
        count = (
            str(progress.done)
            if progress.total is None
            else f"{progress.done}/{progress.total}"
        )
        message = f" {progress.message}" if progress.message else ""
        _say(f"{progress.plugin}: {count}{message}")

    return show


def _check_run_doc(value: str | None) -> Path:
    """The run document's path, or a usage error (exit 2) before anything is opened.

    A run document is never created where the operator did not say, so there is
    no default. ``Path("")`` is ``.``, a folder, so an empty or blank value is
    refused by itself (FM-78); its folder must exist.
    """
    if value is None:
        raise typer.BadParameter(
            "a run needs --run-doc: a run document is never created where you did "
            "not say",
            param_hint="--run-doc",
        )
    if not value.strip():
        raise typer.BadParameter(
            "is empty; name the file to write (an empty path would be the current "
            "folder)",
            param_hint="--run-doc",
        )
    path = Path(value)
    if path.is_dir():
        raise typer.BadParameter(
            f"{path} is a folder; name the file to write", param_hint="--run-doc"
        )
    if not path.parent.is_dir():
        raise typer.BadParameter(
            f"the folder {path.parent} does not exist", param_hint="--run-doc"
        )
    return path


def _list_plugins() -> None:
    """One aligned line per plugin, then a ``✗`` line per entry that did not load.

    A plugin that does not load is a finding, so the listing still exits 0.
    """
    registry = discover_plugins()
    rows = []
    for name, entry in registry.plugins.items():
        plugin = entry.plugin
        requires = ", ".join(capability.__name__ for capability in plugin.requires)
        reads = ", ".join(
            [*plugin.reads, *(f"({slot})" for slot in plugin.reads_optional)]
        )
        rows.append(
            [
                name,
                plugin.summary,
                f"requires {requires or '-'}",
                f"provides {plugin.provides}",
                f"reads {reads or '-'}",
            ]
        )
    for line in _aligned(rows):
        _say(line)
    for failure in registry.failures:
        _fail(f"{failure.name} ({failure.target}): {failure.reason}")
    if not rows and not registry.failures:
        _say(
            "no plugin is registered in the smc.plugins group; if smc was updated "
            "without reinstalling, run pip install -e . in the smc folder (plugins "
            "are registered when it is installed)"
        )


def _outdated(doc: RunDocument, slot: str, read: Any) -> bool:
    """Whether what a stage read of ``slot`` is no longer what the document holds.

    The per-slot form of ``RunDocument.is_stale``, to say which slot: the
    provider is gone, was run again or by another plugin, is itself stale, or an
    optional slot that was empty now has a result.
    """
    current = doc.provider(slot)
    if read is None:
        return current is not None and current.status is StageStatus.done
    return (
        current is None
        or (current.plugin, current.attempt) != (read.plugin, read.attempt)
        or doc.is_stale(current.plugin)
    )


def _stale_lines(before: RunDocument, after: RunDocument) -> list[str]:
    """A warning for each stage that this run made stale: stale after, not before."""
    lines = []
    for plugin, record in sorted(after.stages.items()):
        if before.is_stale(plugin) or not after.is_stale(plugin):
            continue
        for slot, read in record.reads.items():
            if _outdated(after, slot, read):
                lines.append(
                    f"! {plugin} read {slot} from an earlier run; run it again"
                )
    return lines


def _print_result(result: dict[str, Any]) -> None:
    """The result as indented JSON, as a console can show it.

    A lone surrogate from an adapter (FM-05) is in the run document escaped, and
    on a UTF-8 console it would be an ``UnicodeEncodeError``: it prints as a
    replacement character.
    """
    _say(_printable(json.dumps(result, indent=2, ensure_ascii=False)))


def _print_unsaved_result(error: BaseException) -> None:
    """After the ``✗`` line of a run document that could not be written: the result.

    The stage ran, and its result is minutes or hours of work: losing the file
    must not lose what was measured (FM-34).
    """
    if (
        isinstance(error, RunDocumentError)
        and error.record is not None
        and error.record.result is not None
    ):
        _say(
            f"! {error.record.plugin} ran, but its record could not be written; "
            f"its result:"
        )
        _print_result(error.record.result)


def _finish(name: str, path: Path, record: StageRecord, before: RunDocument) -> None:
    """Say how the run ended, once the stand is released."""
    if record.status is StageStatus.stopped:
        # A plugin's own StopRequested: an outcome, recorded, and a stop's exit code.
        _fail(f"{name} stopped")
        raise typer.Exit(code=_EXIT_INTERRUPTED)
    seconds = (
        (record.finished_utc - record.started_utc).total_seconds()
        if record.finished_utc is not None
        else 0.0
    )
    _ok(f"{name} done in {seconds:.1f} s — {path}, stage {name}")
    if record.result is not None:
        _print_result(record.result)
    try:
        after = RunDocument.load_or_new(path)
    except RunDocumentError as exc:
        _say(f"! could not check for stale stages: {exc}")
        return
    for line in _stale_lines(before, after):
        _say(line)


@app.command("run")
def run_command(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(help="The plugin to run; smc run --list shows them."),
    ] = None,
    list_plugins: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List the plugins, what each requires, provides and reads, and "
            "the ones that do not load. Opens no stand.",
        ),
    ] = False,
    show_params: Annotated[
        bool,
        typer.Option(
            "--show-params",
            help="List the plugin's parameters with type, default and bounds. "
            "Opens no stand.",
        ),
    ] = False,
    profile: ProfileOption = DEMO_NAME,
    run_doc: Annotated[
        str | None,
        typer.Option(
            "--run-doc",
            help="The run document: the JSON file that holds the stages of one "
            "experiment. Required for a run; created by the first stage that "
            "writes it, never where you did not say.",
        ),
    ] = None,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="A parameter as key=value; repeat it. A list or a tuple takes "
            "JSON, and null is none. --show-params lists the keys.",
        ),
    ] = None,
    dry_run: DryRunOption = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Continue the stage's last attempt from its checkpoint; the "
            "parameters and the profile must be the same.",
        ),
    ] = False,
) -> None:
    """Run a plugin on the stand and record the outcome in a run document.

    A plugin is an interchangeable tool: smc run --list shows them and what each
    one needs. The stand must have every capability the plugin requires, and the
    stages it reads must be in the run document and current: both are checked,
    and the run document read, before the stand is opened. The result is printed
    after the stand is released, and is in the run document, which is rewritten
    atomically and never overwritten when it cannot be read. A plugin may take
    a while: Ctrl-C stops any move under way and records the stage as stopped.
    """
    with _Reported(ctx, after_report=_print_unsaved_result):
        if list_plugins:
            if name is not None:
                raise typer.BadParameter(
                    "--list takes no plugin name", param_hint="--list"
                )
            _list_plugins()
            return
        if name is None:
            raise typer.BadParameter(
                "name the plugin to run; smc run --list shows the plugins",
                param_hint="NAME",
            )
        entry = load_plugin(name)
        if show_params:
            rows = describe_params(entry.plugin.Params)
            for line in _aligned([list(row) for row in rows]) or ["none"]:
                _say(line)
            return
        try:
            params = parse_params(entry.plugin.Params, param or [])
        except ParamsError as exc:
            # parse_params knows the model, not the plugin: name the plugin here.
            raise ParamsError(name, exc.problems) from None
        path = _check_run_doc(run_doc)
        loaded = Profile.load(profile)
        before = RunDocument.load_or_new(path)
        check_ready(
            entry.plugin,
            before,
            params,
            profile=loaded.microscope.name,
            dry_run=dry_run,
            resume=resume,
        )
        with Microscope.open(loaded, dry_run=dry_run) as microscope:
            if not dry_run:
                _say_stop_refused(microscope)
            record = run_plugin(
                entry.plugin,
                microscope,
                params,
                path,
                resume=resume,
                on_progress=_progress_printer(),
                plugin_version=entry.version,
            )
        _finish(name, path, record, before)


#: What the operator types to get the page's dependencies (FM-36: the brackets are
#: part of the advice, and are printed as text).
_UI_EXTRA_FIX = 'install the ui extra: pip install -e ".[ui]"'

#: The pages ``smc ui`` serves: one, and its name stays accepted (``smc ui plate``),
#: so the notes written for M3 stay true (m5 §1).
_UI_PAGES = ("plate",)


@app.command("ui")
def ui_command(
    ctx: typer.Context,
    page: Annotated[str | None, typer.Argument(hidden=True)] = None,
    profile: ProfileOption = DEMO_NAME,
    run_doc: Annotated[
        str | None,
        typer.Option(
            "--run-doc",
            help="The run document the page reads and its tools write; the same "
            "file smc run takes. Required.",
        ),
    ] = None,
    port: Annotated[
        int,
        typer.Option(
            "--port",
            min=1,
            max=65535,
            help="The port to serve on. The page listens on localhost only: it has "
            "no login, so whoever reaches the port can move the stage.",
        ),
    ] = 5006,
    dry_run: DryRunOption = False,
) -> None:
    """Serve the plate page (needs the ui extra): click wells, run the tools, snap a frame.

    The page opens the stand once and keeps it until Ctrl-C. Every tool button is
    a plugin run with the parameters smc run takes, so each has a command-line
    equivalent, and each tool shows its last result. Ctrl-C stops what is going,
    then the server, then releases the stand.
    """
    # Checked first: a word that names no page is no reason to look at the rest.
    if page is not None and page not in _UI_PAGES:
        _fail(f"unknown page {page}: smc ui serves one page")
        raise typer.Exit(code=_EXIT_REFUSED)
    path = _check_run_doc(run_doc)
    with _Reported(ctx):
        # Text, never the error: nothing here keeps it (FM-98). `import_module`
        # reads `sys.modules` as `import` does, so a test's stand-in is found.
        missing: str | None = None
        try:
            plate_page = importlib.import_module("smc.ui.plate_page")
        except ImportError as exc:
            missing = str(exc)
        if missing is not None:
            _fail(_UI_EXTRA_FIX)
            _say(missing)
            raise typer.Exit(code=_EXIT_ERROR)
        if _port_is_taken(port):
            _fail(f"port {port} is in use; choose another with --port")
            raise typer.Exit(code=_EXIT_REFUSED)
        from smc.ui.controller import PlatePage

        loaded = Profile.load(profile)
        # Read before the stand is opened: an unreadable document is refused as it
        # is, and left untouched.
        RunDocument.load_or_new(path)
        closing: tuple[str, ...] = ()
        ended = True
        with Microscope.open(loaded, dry_run=dry_run) as microscope:
            if not dry_run:
                _say_stop_refused(microscope)
            controller = PlatePage(microscope, path)
            try:
                plate_page.serve(controller, port=port)
            finally:
                # The stop is sent before the wait, so a second Ctrl-C during it
                # still loses nothing, and the stand is released by the `with`
                # whatever ends this (FM-71). A page that `serve` already closed
                # returns at once.
                controller.close()
                closing, ended = controller.close_lines, controller.close_ended
        # After the stand is released: what the stop did and how the run ended, as the
        # page worded them. A run that did not end is a failure, and the stand was
        # released under it (FM-117: never "stopped" by default).
        for line in closing[:-1]:
            _say(line)
        if closing:
            (_say if ended else _fail)(closing[-1])
        if not ended:
            raise typer.Exit(code=_EXIT_ERROR)


def _port_is_taken(port: int) -> bool:
    """Whether ``localhost:<port>`` is held, on any address the page's server will bind.

    Tornado resolves ``localhost`` with ``AF_UNSPEC`` and binds each address it
    gets (``::1`` and ``127.0.0.1``), so a probe of the IPv4 one alone passes a port
    that is held on ``[::1]`` only, and the stand is opened for a server that then
    dies (measured). An address this host cannot bind at all (no IPv6) is not a
    taken port.
    """
    try:
        found = socket.getaddrinfo(
            "localhost", port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except socket.gaierror:
        found = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port))]
    for family, kind, proto, _name, address in found:
        try:
            with socket.socket(family, kind, proto) as probe:
                probe.bind(address)
        except OSError as exc:
            if exc.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                continue
            return True
    return False


def main() -> None:
    """Entry point used by ``python -m smc.cli`` and the ``smc`` console script.

    Reconfigures the streams before ``app()`` runs, not inside the Typer
    callback: Click prints ``--help`` and exits before a callback ever runs,
    so the callback's own fix (kept for ``CliRunner``, which never goes
    through this entry point) never reaches it (FM-42).

    Collects garbage once the command has ended, while the interpreter is
    still whole. What a command left in a reference cycle is then freed here,
    not by the cyclic collector during finalization, where destroying a
    ``CMMCorePlus`` aborted the process when a thread was still inside MMCore
    (#142, FM-92). Click ends a command with ``SystemExit``; its exit code is
    re-raised once the exception and its traceback are gone, which is what
    keeps the command's frames from being alive during the collection. An
    exception that escapes ``app()`` (``--debug``) keeps its traceback, and so
    its frames, alive until the interpreter exits: it is not covered.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            tolerate_unencodable_output(stream)
    exited = False
    code: int | str | None = None  # SystemExit.code's type
    try:
        app()
    except SystemExit as exc:
        exited, code = True, exc.code
    gc.collect()
    if exited:
        sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
