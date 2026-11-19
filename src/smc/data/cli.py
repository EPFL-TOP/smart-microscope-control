"""``smc dataset``: what a scan's run folder holds, read with no stand (algorithms design §3).

The commands of the collaborator's kit that work on a run folder or a dataset
live here, as a ``typer`` sub-app that ``smc.cli`` adds to the root app. This
module never imports ``smc.cli`` (it imports this one), so it has the two
printing helpers of that module for itself: nothing here goes through rich
markup (FM-36), a bracketed path or exception text would lose its brackets, and
nothing is wrapped (FM-37), a one-line refusal must stay one line in a log.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from smc.data import DatasetError, Scan, load_scan

__all__ = ["dataset_app", "summary_lines"]

#: The exit code of a command that refuses its input, as ``smc.cli``'s.
_EXIT_REFUSED = 2

console = Console()

dataset_app = typer.Typer(
    help="Datasets: a scan's run folder, read with no microscope.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

#: The label column of the summary's first block: its longest label, ``place in well``.
_LABEL_WIDTH = len("place in well")

#: The focus states in the order they are told, from the one a scan wants to the one it lacks.
_FOCUS_ORDER = ("locked", "unlocked", "off", "none")


def _say(text: str) -> None:
    """Print a line verbatim: no markup, no highlighting, no emoji codes, no wrapping."""
    console.print(text, markup=False, highlight=False, emoji=False, soft_wrap=True)


def _fail(message: str) -> None:
    """Print ``✗ message``; only the mark is styled."""
    console.print(Text.assemble(("✗ ", "red"), message), soft_wrap=True)


def _aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    """Columns padded to their widest cell, two spaces apart, no trailing blanks."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    ]


def _field(label: str, value: str) -> str:
    return f"{label.ljust(_LABEL_WIDTH)}  {value}".rstrip()


def _focus_text(scan: Scan, well: str) -> str:
    """``locked 109, unlocked 3``: the well's frames per focus state, the states it has."""
    states = [frame.focus for frame in scan.frames(well)]
    return ", ".join(
        f"{state} {states.count(state)}" for state in _FOCUS_ORDER if state in states
    )


def summary_lines(scan: Scan, run_doc: Path | str) -> list[str]:
    """What ``smc dataset show`` prints, one string per line, blank lines included.

    The camera is the scan's (``Scan`` says so), and ``place in well`` counts the
    frames that record their place in their well: a scan written before
    ``well_dx_um`` existed has none, and the line says ``0 of n`` and does not
    hide it (unknown beats guessed).
    """
    result = scan.result
    height, width = scan.image_shape_px
    frames = scan.frames()
    placed = sum(1 for frame in frames if frame.well_dx_um is not None)
    lines = [
        _field("run document", str(run_doc)),
        _field("profile", scan.profile or "unknown"),
        _field("objective", result.objective or "unknown"),
        _field("pixel size", f"{scan.pixel_size_um:.4f} µm/px"),
        _field("image shape", f"{height} x {width} px (height x width)"),
        _field("frames", f"{len(frames)} ({len(result.failed)} fields failed)"),
        _field("place in well", f"{placed} of {len(frames)} frames"),
        "",
    ]
    lines += _aligned(
        [
            ["well", "frames", "focus"],
            *[
                [well, str(len(scan.frames(well))), _focus_text(scan, well)]
                for well in scan.wells
            ],
        ]
    )
    lines.append("")
    lines += _cells_lines(scan)
    return lines


def _cells_lines(scan: Scan) -> list[str]:
    """The cells line, and the cells per class per well when there are cells."""
    if scan.cells is None:
        return [_field("cells", f"none: {scan.cells_absent or 'no reason given'}")]
    cells = scan.cells
    detector = (
        f"detected with {scan.cells_result.model}"
        if scan.cells_result is not None
        else "detector unknown"
    )
    counts: dict[tuple[str, str], int] = {}
    for cell in cells:
        counts[(cell.well, cell.label)] = counts.get((cell.well, cell.label), 0) + 1
    # The detector's classes, zeros shown; a label or a well the result does not
    # know (a scan built in memory) is added after them and never dropped.
    classes = list(scan.cells_result.classes) if scan.cells_result is not None else []
    classes += sorted({label for _, label in counts} - set(classes))
    wells = list(scan.wells)
    wells += sorted({well for well, _ in counts} - set(wells))
    lines = [_field("cells", f"{len(cells)}, {detector}"), ""]
    lines += _aligned(
        [
            ["well", *classes],
            *[
                [well, *[str(counts.get((well, label), 0)) for label in classes]]
                for well in wells
            ],
        ]
    )
    return lines


@dataset_app.command("show")
def show(
    run_doc: Annotated[
        Path,
        typer.Argument(
            metavar="RUN_DOC",
            help="The run document of a scan: its run.json, with the frames "
            "and the cells file where the document says they are.",
        ),
    ],
) -> None:
    """Print what a scan's run folder holds: the camera, the wells, the focus and the cells.

    Reads files only and opens no stand, so it works on a copy of a run folder
    on any PC. Exit 2, with the file and the reason, when the folder holds no
    scan that can be used.
    """
    try:
        scan = load_scan(run_doc)
    except DatasetError as exc:
        _fail(str(exc))
        raise typer.Exit(code=_EXIT_REFUSED) from None
    for line in summary_lines(scan, run_doc):
        _say(line)
