"""A scan's run folder as a self-contained dataset for the lab's share (algorithms design §2).

A collaborator develops a selection algorithm on a laptop with no stand, and the
scan they work on travels as a folder: the run document, the frames and the
cells file at the same relative paths, and a manifest that says what each file
is, so that a copy that lost a byte on the way is found by ``check`` and not by
a wrong result a week later. A dataset *is* a run folder: ``load_scan`` reads it
as it reads the source.

Nothing here imports the hardware layer; ``tests/test_kit_imports.py`` keeps it so.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict

from smc.run.files import retry_on_permission_error

__all__ = ["DatasetFile", "DatasetManifest", "DatasetWriteError", "readme_text"]

#: Read and written a megabyte at a time: a frame is 4 MB or so, and a scan is
#: hundreds of them. A module attribute read at the call, so that a test can
#: make it seven bytes and cut every file into many chunks.
_CHUNK: int = 1 << 20

#: Windows refuses a path of 260 characters or more unless long paths are
#: enabled, and enabling them is the lab PC's choice, not an assumption of ours.
#: Module attributes read at the call, so a test can say it is on Windows with a
#: shorter limit.
MAX_PATH: int = 260
ON_WINDOWS: bool = os.name == "nt"

#: The schema of ``manifest.json`` this smc writes and reads.
MANIFEST_SCHEMA = 1


class DatasetFile(BaseModel):
    """One file of a dataset: its size and the SHA-256 of its bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bytes: int
    sha256: str


class DatasetManifest(BaseModel):
    """What a dataset holds and where it came from; ``manifest.json`` is its dump.

    ``files`` holds every file of the folder but ``manifest.json``, by its
    POSIX path relative to the folder, in the order they were written (the
    frames in the scan's order, the cells file, ``run.json``, ``README.md``).
    ``source`` is the source run document's file name, never its path (FM-50),
    and ``scrubbed`` the JSON pointers of the values of ``run.json`` that the
    export rewrote (``smc.data.scrub``). ``cells_absent`` says why there are no
    cells when there are none. The camera (``pixel_size_um``,
    ``image_shape_px``) is the scan's, not the plan's (FM-124).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = MANIFEST_SCHEMA
    smc_version: str
    exported_utc: AwareDatetime
    source: str
    profile: str
    objective: str | None
    pixel_size_um: float
    image_shape_px: tuple[int, int]
    wells: tuple[str, ...]
    frames: int
    cells: bool
    cells_absent: str | None
    files: dict[str, DatasetFile]
    scrubbed: tuple[str, ...]


class DatasetWriteError(Exception):
    """An export stopped part-way on an I/O error; the message says how far and what to do.

    Not an ``OSError`` and not a ``DatasetError``: the command maps a refusal
    (nothing was written) to exit 2 and this (the folder holds part of the
    dataset) to exit 1, and the cause is chained.
    """


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    """Copy ``source`` to a new file ``destination``; return the bytes written and the SHA-256 of the **source's** bytes.

    The hash is taken of what is read, not of what is written back: a bad write
    (a full disk that lied, a flaky share) is then found by ``check``, where
    hashing the copy afterwards would record the damage as the truth (and read
    the file once more). ``destination`` is opened ``"xb"``: a file that exists
    is never opened for writing. The source is opened first, so that a source
    that cannot be read leaves no empty destination behind.
    """
    digest = hashlib.sha256()
    size = 0
    with retry_on_permission_error(lambda: source.open("rb")) as reader:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as writer:
            while chunk := reader.read(_CHUNK):
                digest.update(chunk)
                writer.write(chunk)
                size += len(chunk)
    return size, digest.hexdigest()


def _sha256_of(path: Path) -> str:
    """The SHA-256 of a file, read once, a chunk at a time.

    ``hashlib.file_digest`` is 3.11 and later, and the floor is 3.10.
    """
    digest = hashlib.sha256()
    with retry_on_permission_error(lambda: path.open("rb")) as reader:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_text(moment: datetime | None) -> str:
    """``2026-11-17 14:32 UTC``, or ``unknown`` for a moment nobody recorded."""
    if moment is None:
        return "unknown"
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def readme_text(
    manifest: DatasetManifest,
    *,
    scanned_utc: datetime | None,
    cells_file: str | None,
    classes: Sequence[str],
) -> str:
    """The ``README.md`` of a dataset: what it is, how it is laid out, what its numbers mean, how to load and check it.

    A collaborator opens this file first, on a laptop that has never seen a
    microscope, so it carries the conventions that are written nowhere else a
    laptop can read (stage axes, the matrix, a pixel size of 0). It is built
    from the manifest *before* the manifest has its ``files``, so it lists no
    hash and no folder name (FM-50), and its fixed text is ASCII (``um``, not
    the micro sign) so that any editor shows it: the objective's name is the
    one place a non-ASCII character can come in, and the file is written as
    UTF-8.

    Args:
        manifest: The manifest as it will be written (``files`` is not read).
        scanned_utc: When the scan started, ``None`` when the run document does
            not say.
        cells_file: The cells file's relative path, ``None`` with no cells.
        classes: The detector's classes, empty with no cells.
    """
    height, width = manifest.image_shape_px
    if manifest.cells and cells_file is not None:
        cells = f"detected, listed in {cells_file}"
        if classes:
            cells += f" (classes: {', '.join(classes)})"
    else:
        cells = f"none: {manifest.cells_absent or 'no reason given'}"
    facts = [
        ("scanned", _utc_text(scanned_utc)),
        (
            "exported",
            f"{_utc_text(manifest.exported_utc)} by smc {manifest.smc_version}",
        ),
        ("profile", manifest.profile or "unknown"),
        ("objective", manifest.objective or "unknown"),
        ("pixel size", f"{manifest.pixel_size_um:.4f} um per pixel"),
        ("image shape", f"{height} x {width} pixels (height x width)"),
        ("wells", f"{', '.join(manifest.wells)} (the order of the scan plan)"),
        ("frames", str(manifest.frames)),
        ("cells", cells),
    ]
    width_of_labels = max(len(label) for label, _ in facts)
    lines = [
        "# A smc scan, as a dataset",
        "",
        "This folder is the run folder of one scan taken with smc, the control layer",
        "of the EPFL-TOP microscopes, copied so that it can be worked on away from",
        "the stand. It needs no microscope.",
        "",
        "## What it is",
        "",
        *[f"{label.ljust(width_of_labels)}  {value}" for label, value in facts],
        "",
        "## Layout",
        "",
        "run.json                        the run document: one record per stage that ran,",
        "                                with its parameters, its result and its error",
        "README.md                       this file",
        "manifest.json                   every other file of the folder, with its size and",
        "                                its hash; written last",
        "scan/<token>/<well>/<name>.tif  one frame per field imaged; run.json lists them",
        "                                in the scan's order, in stages/scan/result/frames",
    ]
    if manifest.cells:
        lines.append(
            "cells/<token>.json              the detected cells, as run.json's cells record names it"
        )
    lines += [
        "",
        "## Units and conventions",
        "",
        "- Lengths are in um (micrometres), times in ms or s, angles in degrees: a name",
        "  says its unit (x_um, exposure_ms, rotation_deg, box_px).",
        "- A frame's x_um and y_um are the stage's readback after the move, in the",
        "  stage's axes: the point that was imaged at the centre of the frame.",
        "- well_dx_um and well_dy_um say where a frame is in its well: the readback",
        "  minus the well's centre, in stage axes. A frame that has none was taken",
        "  before smc recorded them: unknown, not zero.",
        "- The camera matrix M (matrix_um_per_px in the scan's result) turns a pixel",
        "  offset into a stage offset: (dx_um, dy_um) = M @ (d_col, d_row), where",
        "  (d_col, d_row) is the offset in pixels from the frame's centre, as",
        "  smc.plugins.geometry states it.",
        "- A frame is an array of (height, width) pixels. A cell's box_px is",
        "  (row0, col0, row1, col1) in the pixels of its frame, and its frame is an",
        "  index into the scan's frames.",
        "- A pixel size of 0 is unknown: smc reports it and never defaults it.",
        "",
        "## Load it",
        "",
        "With smc installed (the smart-microscope-control repository), from this folder:",
        "",
        "    from smc.data import load_scan",
        "",
        '    scan = load_scan("run.json")',
        "    scan.wells                      # the wells, in the plan's order",
        '    frame = scan.frames("A1")[0]',
        "    pixels = scan.image(frame)      # a numpy array, (height, width)",
        "    scan.cells                      # the cells, or None: scan.cells_absent says why",
        "",
        "Micro-Manager is not needed.",
        "",
        "## Check it",
        "",
        "    smc dataset check <this folder>",
        "",
        "reads every file that manifest.json lists and compares its size and its hash,",
        "then loads the scan. Exit 0: all match. Exit 1: a file is missing or differs,",
        "each one named. Exit 2: this is not a dataset, or smc refuses the scan. Run it",
        "after every copy or unzip, before anything else.",
        "",
        "Work on a copy: a tool that writes a stage into this folder's run.json makes",
        "check report it changed. Files that manifest.json does not list (your own",
        "outputs) are ignored.",
        "",
        "## What was left out",
        "",
        "Only the scan's frames and the cells file were copied. run.json still holds",
        "the records of the other stages (plate registration, scan plan, and so on),",
        "but the files those stages wrote (reports, drawings, exports) are not here.",
        "",
        "Paths in the parameters, results and errors of run.json were cut to their",
        "last component, so that no account or host name travels with the dataset;",
        'manifest.json lists the values it rewrote under "scrubbed".',
        "",
    ]
    return "\n".join(lines)
