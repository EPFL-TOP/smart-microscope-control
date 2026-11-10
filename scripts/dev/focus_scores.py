"""Score Z-stacks that ``focus-on-cells`` kept, with every focus score (m3 section 21.8, #341).

Each folder is one stack: the TIFFs in it whose JSON metadata has ``z_um``
(the tool's ``<k>.tif`` frames; ``best.tif`` is left out). For each stack it
prints one row per plane (Z and the four scores) and, per score, the outcome
and the refined Z of ``find_peak``, so that the default score and the sweep's
range and step are chosen on real 40x stacks before the scan uses the step::

    python scripts/dev/focus_scores.py local/cell-focus/* --csv scores.csv --plot plots

``--csv`` writes every row with its folder; ``--plot`` writes one PNG per stack
(each score divided by its maximum, against Z) when ``matplotlib`` imports, and
says so and goes on when it does not. It imports
``smc.plugins.cell_focus.algorithm``, so this script and the tool score alike.
Output is ASCII and files are UTF-8 (a Windows console is not), and nothing here
opens a stand.

A folder argument may be a pattern (``local/cell-focus/*``): the script expands
it, as cmd.exe and PowerShell do not. A frame Windows refuses to read for a
moment is read again; one that stays unreadable skips its stack, and a TIFF
without ``z_um`` is left out and counted.

Exit codes: 0 when at least one stack was scored and every output asked for was
written, 1 when no stack was scored or an output could not be written (each is
explained on stderr), 2 for a usage error.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
import numpy.typing as npt
import tifffile

from smc.plugins.cell_focus.algorithm import ScoreName, find_peak, score_frame
from smc.run.files import retry_on_permission_error

SCORES: tuple[ScoreName, ...] = get_args(ScoreName)
#: The tool's frame at the best plane: a repeat of a plane, not a plane of the stack.
BEST = "best.tif"


@dataclass(frozen=True)
class Plane:
    """One frame of a stack: its Z readback and its score by each name."""

    z_um: float
    scores: dict[str, float]


def read_stack(folder: Path) -> tuple[list[Plane], str]:
    """The stack's planes in increasing Z and a note on what was left out; or none and why.

    A TIFF without a finite ``z_um`` in its metadata is not a plane of a stack
    (a frame of another tool), and the stack says how many it left out. A
    plane the OS will not read is not left out: a curve with a hole in it would
    choose a score on planes that are not all there, so the stack is skipped.
    Each read is retried while Windows refuses it for a moment (FM-24:
    antivirus or a backup agent on a file written minutes ago).
    """
    if not folder.is_dir():
        return [], f"{folder}: not a folder"
    planes: list[Plane] = []
    left_out = 0
    for path in sorted(folder.glob("*.tif")):
        if path.name == BEST:
            continue
        unreadable: OSError | None = None
        try:
            z_um = _z_of(path)
            frame: npt.NDArray[Any] | None = (
                None
                if z_um is None
                else np.asarray(
                    retry_on_permission_error(lambda path=path: tifffile.imread(path))
                )
            )
        except OSError as exc:
            unreadable = exc
        if unreadable is not None:
            reason = unreadable.strerror or unreadable
            return [], f"{folder}: {path.name} cannot be read ({reason}): skipped"
        if z_um is None or frame is None:
            left_out += 1
            continue
        planes.append(Plane(z_um, {name: score_frame(frame, name) for name in SCORES}))
    note = (
        f"{left_out} TIFFs without z_um in their metadata left out" if left_out else ""
    )
    if len(planes) < 3:
        return [], (
            f"{folder}: {len(planes)} frames with z_um in their metadata, a stack "
            f"needs 3: skipped" + (f" ({note})" if note else "")
        )
    planes.sort(key=lambda plane: plane.z_um)
    if len({plane.z_um for plane in planes}) != len(planes):
        return [], f"{folder}: two frames have the same z_um: skipped"
    return planes, note


def _z_of(path: Path) -> float | None:
    """The ``z_um`` of a TIFF's JSON metadata, or ``None`` when it has none, it is not a finite number, or the file is not a TIFF.

    Raises:
        OSError: The OS would not read the file, after the retries.
    """

    def metadata() -> Any:
        with tifffile.TiffFile(path) as tiff:
            return tiff.shaped_metadata

    try:
        stored = retry_on_permission_error(metadata)
    except ValueError:
        # Not a TIFF tifffile can parse (its TiffFileError is a ValueError).
        return None
    if not stored:
        return None
    value = dict(stored[0]).get("z_um")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def report(folder: Path, planes: Sequence[Plane], out: Any) -> None:
    """One row per plane, then each score's outcome and refined Z."""
    out.write(f"{folder}\n")
    out.write("  " + "z_um".rjust(10) + "".join(f"{n:>20}" for n in SCORES) + "\n")
    for plane in planes:
        cells = "".join(f"{plane.scores[n]:>20.6g}" for n in SCORES)
        out.write(f"  {plane.z_um:>10.2f}{cells}\n")
    z = [plane.z_um for plane in planes]
    for name in SCORES:
        peak = find_peak(z, [plane.scores[name] for plane in planes])
        at = f"{peak.z_um:.2f} um" if peak.z_um is not None else "kept the centre"
        best = z[peak.best_index]
        out.write(
            f"  {name}: {peak.outcome}, best plane {best:.2f} um, refined: {at}\n"
        )


def write_csv(path: Path, stacks: Sequence[tuple[Path, Sequence[Plane]]]) -> None:
    """Every plane of every stack scored, one row each, with its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["folder", "z_um", *SCORES])
        for folder, planes in stacks:
            for plane in planes:
                writer.writerow(
                    [folder.as_posix(), plane.z_um, *(plane.scores[n] for n in SCORES)]
                )


def write_plots(
    folder: Path, stacks: Sequence[tuple[Path, Sequence[Plane]]], err: Any
) -> tuple[int, int]:
    """One PNG per stack, each score divided by its maximum; how many were written, and how many could not be.

    Without matplotlib nothing is written and that is said, not counted as a
    failure. A PNG the OS refuses is said and the next is tried.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        err.write("matplotlib is not installed: no plot written\n")
        return 0, 0
    written = 0
    problems = 0
    for index, (source, planes) in enumerate(stacks):
        z = [plane.z_um for plane in planes]
        figure, axes = plt.subplots(figsize=(6.0, 4.0))
        for name in SCORES:
            values = [plane.scores[name] for plane in planes]
            top = max(values)
            axes.plot(z, [v / top if top > 0 else 0.0 for v in values], label=name)
        axes.set_xlabel("Z (um)")
        axes.set_ylabel("score / its maximum")
        axes.set_title(source.as_posix())
        axes.legend()
        # The folders of the tool are named by their token; the index keeps two
        # stacks of the same name apart.
        target = folder / f"{index:02d}-{source.name}.png"
        problem: OSError | None = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=100)
        except OSError as exc:
            problem = exc
        plt.close(figure)
        if problem is not None:
            err.write(
                f"{target} could not be written ({problem.strerror or problem})\n"
            )
            problems += 1
        else:
            written += 1
    return written, problems


def main(argv: Sequence[str] | None = None) -> int:
    """Score each stack, print it, and write the CSV and the plots asked for."""
    parser = argparse.ArgumentParser(
        description="Score the Z-stacks focus-on-cells kept, with every focus score."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="one stack per folder; a pattern (local/cell-focus/*) is expanded here",
    )
    parser.add_argument("--csv", type=Path, help="write every row to this CSV file")
    parser.add_argument("--plot", type=Path, help="write one PNG per stack here")
    args = parser.parse_args(argv)

    stacks: list[tuple[Path, Sequence[Plane]]] = []
    for folder in expand(args.folders):
        planes, why = read_stack(folder)
        if not planes:
            sys.stderr.write(why + "\n")
            continue
        if why:
            sys.stderr.write(f"{folder}: {why}\n")
        report(folder, planes, sys.stdout)
        stacks.append((folder, planes))
    if not stacks:
        sys.stderr.write("no stack was scored\n")
        return 1
    failed = False
    if args.csv is not None:
        problem: OSError | None = None
        try:
            write_csv(args.csv, stacks)
        except OSError as exc:
            problem = exc
        if problem is not None:
            # A CSV that Excel holds open is refused on Windows: say it and
            # still write the plots.
            sys.stderr.write(
                f"{args.csv} could not be written ({problem.strerror or problem})\n"
            )
            failed = True
        else:
            sys.stdout.write(f"wrote {args.csv}\n")
    if args.plot is not None:
        written, problems = write_plots(args.plot, stacks, sys.stderr)
        if written:
            sys.stdout.write(f"wrote {written} plots in {args.plot}\n")
        failed = failed or problems > 0
    return 1 if failed else 0


def expand(given: Sequence[str]) -> list[Path]:
    """The folders named, with a pattern expanded here.

    cmd.exe and PowerShell pass ``local/cell-focus/*`` to a program as it is
    typed, and Python does not expand it on Windows: the script does, so the
    documented command works in every shell. A name that exists is taken as it
    is, even with ``[`` in it; a pattern that matches nothing stays, and is
    reported as not a folder.
    """
    folders: list[Path] = []
    for text in given:
        matches = (
            sorted(glob.glob(text))
            if not Path(text).exists() and any(c in text for c in "*?[")
            else []
        )
        if matches:
            folders.extend(Path(match) for match in matches)
        else:
            folders.append(Path(text))
    return folders


if __name__ == "__main__":
    sys.exit(main())
