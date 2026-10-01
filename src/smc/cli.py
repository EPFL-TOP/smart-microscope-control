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
import platform
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TextIO

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from smc import __version__
from smc.hardware import core as core_mod
from smc.hardware.capabilities import XY
from smc.hardware.errors import HardwareError, SafetyRefusedError

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


def _configure_logging(level: int) -> None:
    """Send the ``smc`` loggers to stderr at ``level``; the root logger is left alone.

    Records still propagate, so an application or ``pytest`` that configured
    the root logger keeps receiving them.
    """
    logger = logging.getLogger("smc")
    for handler in list(logger.handlers):
        if isinstance(handler, _CliLogHandler):
            logger.removeHandler(handler)
    handler = _CliLogHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


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
    _configure_logging(
        logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
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
def doctor(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="A Micro-Manager .cfg to test-load. Default: the demo devices.",
        ),
    ] = None,
) -> None:
    """Check that this machine can drive a microscope, or at least the simulator.

    Exits non-zero when something is missing, so it can gate a session script.
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

    target = str(config) if config else "demo configuration"
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
