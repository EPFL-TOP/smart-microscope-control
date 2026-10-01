"""``smc`` — the command line is the reference interface of this project.

Every plugin and every hardware capability is reachable from here before it
is reachable from any graphical interface (ADR-0006), because a command line
is scriptable, testable and works over SSH to a microscope PC.
"""

from __future__ import annotations

import codecs
import contextlib
import logging
import math
import os
import platform
import secrets
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TextIO

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from smc import __version__
from smc.hardware import core as core_mod
from smc.hardware.capabilities import XY, Camera, XYStage, ZStage
from smc.hardware.errors import HardwareError, ProfileError, SafetyRefusedError
from smc.hardware.microscope import Microscope
from smc.hardware.profile import DEMO_NAME, Profile, list_profiles, search_paths
from smc.hardware.roles import Role

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


@contextlib.contextmanager
def _reported(ctx: typer.Context, *, force_hint: str = "") -> Iterator[None]:
    """Turn what a command body raises into one ``✗`` line and an exit code.

    Wrapped around the ``with Microscope.open(...)`` block, so the stand is
    released before the line is printed. Ctrl-C is the ``KeyboardInterrupt``
    the Executor already turned into a stop of any move under way (FM-17);
    there is no signal handler (FM-70).

    Args:
        ctx: The command's context; ``--debug`` lets every error through,
            so Typer prints the traceback.
        force_hint: What the operator types to force a forceable refusal; it
            replaces the API's ``how_to_force`` (``force=True``).
    """
    try:
        yield
    except (typer.Exit, typer.Abort, typer.BadParameter):
        raise
    except (Exception, KeyboardInterrupt) as exc:
        settings = ctx.find_object(_Settings)
        if settings is not None and settings.debug:
            raise
        raise typer.Exit(code=_report(exc, force_hint)) from None


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
        _fail(
            "interrupted — any move under way was stopped; read the position "
            "(smc stage get, smc z get) before the next move"
        )
        return _EXIT_INTERRUPTED
    _fail(
        f"unexpected {type(exc).__name__}: {exc} — run it again as smc --debug … "
        "for the traceback, and file an issue"
    )
    return _EXIT_ERROR


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
    with _reported(ctx):
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
    except (OSError, ValueError) as exc:
        # A file that is not UTF-8 raises UnicodeDecodeError (a ValueError)
        # past the loader; it is one more profile that does not load.
        message = f"{type(exc).__name__}: {exc}"
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
    with _reported(ctx), Microscope.open(profile) as microscope:
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
        with _reported(ctx), Microscope.open(profile) as microscope:
            loaded = microscope.profile
            name = loaded.microscope.name
            _say(f"Profile {name} ({loaded.source or 'built in'})")
            _print_roles(microscope)
            # The status line shows a failed read as "?" and nothing more;
            # the reason is in the errors, which decide the exit code.
            errors = microscope.state().errors
            _say(microscope.describe())
        if errors:
            for error in errors:
                _fail(error)
            raise typer.Exit(code=_EXIT_ERROR)
        _ok(f"{name} loads and answers.")
        return

    target = str(config)
    try:
        with core_mod.opened(config) as core:
            roles = {
                "camera": core.getCameraDevice(),
                "xy stage": core.getXYStageDevice(),
                "focus": core.getFocusDevice(),
                "autofocus": core.getAutoFocusDevice(),
                "shutter": core.getShutterDevice(),
            }
            rt = Table(title=f"Core roles — {target}", show_header=False, box=None)
            for role, device in roles.items():
                rt.add_row(role, device or "[yellow]— not set —[/yellow]")
            if roles["xy stage"]:
                x, y = core.getXYPosition()
                rt.add_row("xy position", f"({x:.1f}, {y:.1f}) µm")
            console.print(rt)
    except core_mod.CoreError as exc:
        _fail(str(exc))
        raise typer.Exit(code=1) from None
    _ok(f"{target} loads and answers.")


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

    if not math.isfinite(timeout_s):  # click's range check lets nan through
        _fail(f"--timeout-s must be a number of seconds, not {timeout_s}")
        raise typer.Exit(code=2)
    # --out is checked before the survey and written before the report is
    # printed: a broken pipe or Ctrl-C while printing must not lose the files.
    if out is not None:
        try:
            report_mod.prepare(out)
        except OSError as exc:
            _fail(f"cannot write to {out}: {exc}")
            raise typer.Exit(code=1) from None
    if probe_adapter:
        console.print(
            f"[yellow]![/yellow] Probing {probe_adapter}: the stand must be "
            "powered and its vendor software closed."
        )
    with console.status("Surveying this PC…"):
        inv = inventory(
            all_adapters=all_adapters,
            probe_adapter=probe_adapter,
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
        raise typer.Exit(code=1)
    if written is not None:
        json_path, text_path = written
        _ok(f"wrote {json_path} and {text_path.name}")


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
        _reported(ctx, force_hint=force_hint),
        Microscope.open(profile, dry_run=dry_run) as microscope,
    ):
        stage = microscope.require(XYStage)
        landed = move(stage)
        # In dry-run the move returns the commanded target; the stage itself
        # is read again to show that it did not move.
        here = stage.position_um() if dry_run else None
    if here is None:
        _say(_xy(landed))
    else:
        _say(f"[dry-run] would move to {_xy(landed)}; the stage is at {_xy(here)}")


def _move_z(
    ctx: typer.Context, profile: str, dry_run: bool, move: Callable[[ZStage], float]
) -> None:
    """Open the stand, run ``move`` on its focus drive, print where it landed."""
    with _reported(ctx), Microscope.open(profile, dry_run=dry_run) as microscope:
        drive = microscope.require(ZStage)
        landed = move(drive)
        here = drive.position_um() if dry_run else None
    if here is None:
        _say(_z(landed))
    else:
        _say(f"[dry-run] would move to {_z(landed)}; the drive is at {_z(here)}")


@stage_app.command("get")
def stage_get(ctx: typer.Context, profile: ProfileOption = DEMO_NAME) -> None:
    """Print where the XY stage reports it is, in µm."""
    with _reported(ctx), Microscope.open(profile) as microscope:
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
    with _reported(ctx), Microscope.open(profile) as microscope:
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
    dry_run: DryRunOption = False,
    profile: ProfileOption = DEMO_NAME,
) -> None:
    """Move the focus drive by a distance in µm.

    There is no jog guard on Z: only the profile's safety.z_soft_limits_um
    bound the target.
    """
    _move_z(ctx, profile, dry_run, lambda drive: drive.move_by_um(dz_um))


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
    leave every capture readable by its owner only. A partial file is
    removed when the write fails. When the rename fails (Windows refuses to
    replace a TIFF that Fiji holds open), ``out`` stays as it was and the
    new file is kept: it holds a frame that was taken.

    Raises:
        OSError: The folder cannot be written, or ``out`` cannot be replaced;
            the message then names the file that was kept.
    """
    partial = out.with_name(f".{out.name}.{secrets.token_hex(4)}.part")
    # O_EXCL: never write into a file that something else created.
    os.close(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    try:
        write(partial)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink()
        raise
    try:
        os.replace(partial, out)
    except OSError as exc:
        raise OSError(f"{exc}; the new file is kept as {partial}") from exc


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
    with _reported(ctx):
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
    raise typer.Exit(code=_EXIT_ERROR)


def main() -> None:
    """Entry point used by ``python -m smc.cli`` and the ``smc`` console script.

    Reconfigures the streams before ``app()`` runs, not inside the Typer
    callback: Click prints ``--help`` and exits before a callback ever runs,
    so the callback's own fix (kept for ``CliRunner``, which never goes
    through this entry point) never reaches it (FM-42).
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            tolerate_unencodable_output(stream)
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
