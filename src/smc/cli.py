"""``smc`` — the command line is the reference interface of this project.

Every plugin and every hardware capability is reachable from here before it
is reachable from any graphical interface (ADR-0006), because a command line
is scriptable, testable and works over SSH to a microscope PC.
"""

from __future__ import annotations

import codecs
import platform
import sys
from pathlib import Path
from typing import Annotated, TextIO

import typer
from rich.console import Console
from rich.table import Table

from smc import __version__
from smc.hardware import core as core_mod

app = typer.Typer(
    name="smc",
    help="Control several microscopes through one layer; run interchangeable tools.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


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
def _main() -> None:
    """Control several microscopes through one layer; run interchangeable tools."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            tolerate_unencodable_output(stream)


@app.command()
def version() -> None:
    """Print the versions that matter when reporting a problem."""
    import pymmcore_plus

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
        console.print(f"[red]✗[/red] {core_mod.INSTALL_HINT}")
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
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from None
    console.print(f"[green]✓[/green] {target} loads and answers.")


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
        typer.Option("--timeout-s", help="How long the adapter child process may run."),
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
    for item in report_mod.renderables(inv):
        console.print(item)
    if out is not None:
        json_path, text_path = report_mod.write(inv, out)
        console.print(f"[green]✓[/green] wrote {json_path} and {text_path.name}")


def main() -> None:
    """Entry point used by ``python -m smc.cli``."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
