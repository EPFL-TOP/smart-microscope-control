"""``smc dataset``: what a scan's run folder holds, read with no stand (algorithms design §3).

The commands of the collaborator's kit that work on a run folder or a dataset
live here, as a ``typer`` sub-app that ``smc.cli`` adds to the root app. This
module never imports ``smc.cli`` (it imports this one), so it has the printing
helpers of that module for itself: nothing here goes through rich markup
(FM-36), a bracketed path or exception text would lose its brackets, and
nothing is wrapped (FM-37), a one-line refusal must stay one line in a log.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from smc.data import DatasetError, Scan, load_scan
from smc.data.dataset import (
    DatasetCheck,
    DatasetManifest,
    DatasetWriteError,
    check_dataset,
    export_dataset,
)

__all__ = ["check_lines", "dataset_app", "export_lines", "summary_lines"]

#: The exit code of a command that refuses its input, as ``smc.cli``'s.
_EXIT_REFUSED = 2

#: The exit codes of the two commands that say more than "refused": ``export``
#: stopped part-way on an I/O error (the folder holds part of the dataset), and
#: ``check`` found a file that is missing or differs.
_EXIT_PARTIAL = 1
_EXIT_DIFFERS = 1

#: The progress printer's pace: the first call, then one a second, and always the last.
_PROGRESS_INTERVAL_S = 1.0

#: How many files ``check`` names when many differ; the rest are counted.
_PROBLEMS_NAMED = 20

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


def _ok(message: str) -> None:
    """Print ``✓ message``; only the mark is styled."""
    console.print(Text.assemble(("✓ ", "green"), message), soft_wrap=True)


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


def _progress_printer(
    label: str, clock: Callable[[], float] = time.monotonic
) -> Callable[[int, int], None]:
    """A progress callback that prints ``label: done/total``, at most a line a second.

    The first call is printed, then a call at least a second after the last
    printed one, and always the last (``done == total``): a short copy prints
    its first and last lines, a long one a line a second. Plain lines, as
    ``smc run``'s: a ``rich`` bar on a redirected stream draws nothing and leaves
    an empty line (measured, rich 15). ``smc.cli``'s printer is not imported:
    that module imports this one.
    """
    last_printed: float | None = None

    def show(done: int, total: int) -> None:
        nonlocal last_printed
        now = clock()
        if (
            last_printed is not None
            and done != total
            and now - last_printed < _PROGRESS_INTERVAL_S
        ):
            return
        last_printed = now
        _say(f"{label}: {done}/{total}")

    return show


def _megabytes(manifest: DatasetManifest) -> str:
    """The size of what the manifest lists, as ``12.3 MB`` (decimal: ``/1e6``, as a disk is sold)."""
    total = sum(file.bytes for file in manifest.files.values())
    return f"{total / 1e6:.1f} MB"


def _cells_file_of(manifest: DatasetManifest) -> str | None:
    """The cells file's path, from the order the manifest lists its files in.

    The frames come first, in the scan's order, then the cells file when there
    is one, then ``run.json`` and ``README.md`` (``DatasetManifest``).
    """
    if not manifest.cells:
        return None
    names = list(manifest.files)
    return names[manifest.frames] if manifest.frames < len(names) else None


def _cells_text(manifest: DatasetManifest, count: int | None) -> str:
    """``<n> in <file>`` with cells, ``none: <why>`` without; the count is left out when it is not known."""
    if not manifest.cells:
        return f"none: {manifest.cells_absent or 'no reason given'}"
    file = _cells_file_of(manifest) or "the cells file"
    return file if count is None else f"{count} in {file}"


def export_lines(manifest: DatasetManifest, *, cells: int | None = None) -> list[str]:
    """What ``smc dataset export`` prints after its first line, one string per line.

    ``cells`` is the number of cells in the exported file: the manifest records
    that there are cells and in which file, not how many, so the command reads
    the count from the dataset it just wrote and a caller that does not have it
    leaves it out. ``scrubbed`` says how many values of ``run.json`` the export
    rewrote; their pointers are in ``manifest.json``, which is where a person
    who wants to see them is sent.
    """
    scrubbed = len(manifest.scrubbed)
    rewritten = f"{scrubbed} value{'' if scrubbed == 1 else 's'}"
    if scrubbed:
        rewritten += ", listed in manifest.json"
    return _aligned(
        [
            ["frames", str(manifest.frames)],
            ["wells", ", ".join(manifest.wells)],
            ["cells", _cells_text(manifest, cells)],
            ["size", f"{_megabytes(manifest)} in {len(manifest.files)} files"],
            ["scrubbed", rewritten],
        ]
    )


def check_lines(result: DatasetCheck) -> list[str]:
    """What ``smc dataset check`` prints after its first line when every file matched."""
    manifest = result.manifest
    count = (
        len(result.scan.cells)
        if result.scan is not None and result.scan.cells is not None
        else None
    )
    return _aligned(
        [
            ["profile", manifest.profile or "unknown"],
            ["frames", f"{manifest.frames} in wells {', '.join(manifest.wells)}"],
            ["cells", _cells_text(manifest, count)],
        ]
    )


def _cell_count(out: Path, manifest: DatasetManifest) -> int | None:
    """How many cells the dataset just written holds, or ``None`` when it has none or cannot say.

    Read with the loader the collaborator will use, so that the line also says
    the dataset loads; an answer that cannot be had is left out of the line and
    does not turn a finished export into a failure.
    """
    if not manifest.cells:
        return None
    try:
        cells = load_scan(out / "run.json").cells
    except DatasetError:
        return None
    return None if cells is None else len(cells)


@dataset_app.command("export")
def export(
    run_doc: Annotated[
        Path,
        typer.Argument(
            metavar="RUN_DOC",
            help="The run document of the scan: its run.json, with the frames "
            "and the cells file where the document says they are.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            metavar="DIR",
            help="The folder to write the dataset to: it must not exist yet or "
            "must be empty, and its parents are made.",
        ),
    ],
) -> None:
    """Write a scan's run folder as a dataset to hand to a collaborator, and say what is in it.

    Copies the frames and the cells file, writes the run document with the
    paths in it cut to their last name, a README and a manifest of every file
    with its size and hash. Nothing is overwritten and nothing is deleted. Reads
    files only and opens no stand. Exit 2, with the reason, when nothing was
    written; exit 1 when an I/O error stopped it part-way (the message says what
    to remove).
    """
    try:
        manifest = export_dataset(run_doc, out, progress=_progress_printer("copied"))
    except DatasetError as exc:
        _fail(str(exc))
        raise typer.Exit(code=_EXIT_REFUSED) from None
    except DatasetWriteError as exc:
        _fail(str(exc))
        raise typer.Exit(code=_EXIT_PARTIAL) from None
    _ok(f"dataset written to {out}")
    for line in export_lines(manifest, cells=_cell_count(out, manifest)):
        _say(line)


@dataset_app.command("check")
def check(
    directory: Annotated[
        Path,
        typer.Argument(
            metavar="DIR",
            help="The dataset folder: the one with a manifest.json in it.",
        ),
    ],
) -> None:
    """Check a dataset folder against its manifest: every file's size and hash, then that the scan loads.

    Run it after every copy or unzip. Files the manifest does not list are
    ignored. Exit 0 when everything matches; exit 1 when a file is missing or
    differs (each is named, the first twenty); exit 2 when there is no
    manifest, the manifest cannot be read or the scan cannot be loaded.
    """
    try:
        result = check_dataset(directory, progress=_progress_printer("checked"))
    except DatasetError as exc:
        _fail(str(exc))
        raise typer.Exit(code=_EXIT_REFUSED) from None
    except Exception as exc:
        # Not a verdict: an uncaught exception would exit 1, which is "a file
        # differs" (FM-82), so whatever `check` did not expect is a refusal.
        _fail(f"unexpected {type(exc).__name__}: {exc}")
        raise typer.Exit(code=_EXIT_REFUSED) from None
    files = len(result.manifest.files)
    if result.problems:
        for problem in result.problems[:_PROBLEMS_NAMED]:
            _fail(problem)
        if len(result.problems) > _PROBLEMS_NAMED:
            _fail(f"and {len(result.problems) - _PROBLEMS_NAMED} more")
        _fail(
            f"{len(result.problems)} of {files} files are missing or changed: "
            f"copy the dataset again, then check again"
        )
        raise typer.Exit(code=_EXIT_DIFFERS)
    _ok(f"{files} files match manifest.json ({_megabytes(result.manifest)})")
    for line in check_lines(result):
        _say(line)
