"""``smc`` — the command line is the reference interface of this project.

Every plugin and every hardware capability is reachable from here before it
is reachable from any graphical interface (ADR-0006), because a command line
is scriptable, testable and works over SSH to a microscope PC.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Annotated

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


def main() -> None:
    """Entry point used by ``python -m smc.cli``."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
