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
import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from smc import __version__
from smc.data.scan import DatasetError, Scan, load_scan
from smc.data.scrub import scrub_document
from smc.run.files import (
    resolve_in_run,
    retry_on_permission_error,
    temporary_path,
    write_atomically,
)

__all__ = [
    "DatasetCheck",
    "DatasetFile",
    "DatasetManifest",
    "DatasetWriteError",
    "check_dataset",
    "export_dataset",
    "readme_text",
]

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

#: The three files the export writes itself, after the copies, in this order:
#: the manifest last, so that a folder without one is an export that did not
#: finish (FM-106).
_RUN_DOCUMENT = "run.json"
_README = "README.md"
_MANIFEST = "manifest.json"

#: What the free space must exceed the files by: the three generated files and
#: the file system's own overhead.
_HEADROOM_BYTES = 1_000_000

#: How many missing frames a refusal names; the rest are counted.
_MISSING_NAMED = 5


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
        # The scan's own first well: a scan of other wells than A1 must run this too.
        f'    frame = scan.frames("{manifest.wells[0] if manifest.wells else "A1"}")[0]',
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
        "last component, which removes the account and host names they held; one",
        "exception is known: a folder name with a blank in it, inside a sentence, is",
        'cut at the blank. manifest.json lists the values it rewrote under "scrubbed".',
        "",
    ]
    return "\n".join(lines)


def _json_bytes(model: BaseModel) -> bytes:
    """A model as the run document is written: ASCII, indent 2, a final newline.

    The arguments of ``write_stage``: ASCII so that no console or code page can
    mangle the file (a lone surrogate and a micro sign are escaped), and
    ``allow_nan=False`` as a seatbelt, since ``mode="json"`` already turns a
    non-finite number into ``null``.
    """
    text = json.dumps(
        model.model_dump(mode="json"), indent=2, ensure_ascii=True, allow_nan=False
    )
    return (text + "\n").encode("ascii")


def _check_out(out: Path) -> Path:
    """``out`` resolved, or a refusal: the export never overwrites and never deletes.

    A folder that does not exist is made, and an empty one is used; anything
    else (a file, a folder with something in it) is the operator's, and is
    named back to them with what to do.
    """
    try:
        root = out.resolve()
        if out.is_dir():
            entries = sum(1 for _ in out.iterdir())
            if entries:
                raise DatasetError(
                    f"{out}: exists and is not empty ({entries} entries); export "
                    f"never overwrites or deletes: name a folder that does not "
                    f"exist yet, or is empty"
                )
        elif out.exists():
            raise DatasetError(
                f"{out}: exists and is a file; name a folder that does not exist "
                f"yet, or is empty"
            )
    except OSError as exc:
        # `resolve` and `stat` on a share that does not answer (FM-116).
        raise DatasetError(
            f"{out}: cannot be examined ({exc.strerror or exc}); name a folder "
            f"on a disk that answers"
        ) from exc
    return root


def _frame_sources(run_doc: Path, scan: Scan) -> list[tuple[str, Path, int]]:
    """Each frame's relative path, its source file and its size; a refusal naming the ones that are missing, or that the disk will not say.

    A frame whose path cannot be resolved inside the run folder (a ``..``, an
    absolute path from another PC) counts as missing, and is never read. A
    frame that the file system *cannot answer for* (a share that dropped:
    ``resolve_in_run`` turns the ``OSError`` of ``resolve`` into a ``ValueError``
    chained to it, and ``stat`` raises it itself) is told apart from a missing
    one (FM-116): the operator is not told that a night's scan was moved or
    deleted when retrying would work.
    """
    found: list[tuple[str, Path, int]] = []
    missing: list[str] = []
    unreadable: list[tuple[str, str, Exception]] = []
    for frame in scan.result.frames:
        try:
            source = resolve_in_run(scan.run_dir, frame.file)
            size = source.stat().st_size if source.is_file() else None
        except FileNotFoundError:
            size = None
        except ValueError as exc:
            if isinstance(exc.__cause__, OSError):
                cause = exc.__cause__
                unreadable.append((frame.file, str(cause.strerror or cause), exc))
                continue
            size = None
        except OSError as exc:
            unreadable.append((frame.file, str(exc.strerror or exc), exc))
            continue
        if size is None:
            missing.append(frame.file)
            continue
        found.append((frame.file, source, size))
    if unreadable:
        named = ", ".join(name for name, _, _ in unreadable[:_MISSING_NAMED])
        more = (
            f" (and {len(unreadable) - _MISSING_NAMED} more)"
            if len(unreadable) > _MISSING_NAMED
            else ""
        )
        raise DatasetError(
            f"{run_doc}: {len(unreadable)} of {len(scan.result.frames)} frame "
            f"files cannot be read now ({unreadable[0][1]}): {named}{more}; the "
            f"run folder is on a disk or a share that does not answer: try again "
            f"when it does"
        ) from unreadable[0][2]
    if missing:
        named = ", ".join(missing[:_MISSING_NAMED])
        more = (
            f" (and {len(missing) - _MISSING_NAMED} more)"
            if len(missing) > _MISSING_NAMED
            else ""
        )
        raise DatasetError(
            f"{run_doc}: {len(missing)} of {len(scan.result.frames)} frame files "
            f"are missing from the run folder: {named}{more}; the scan's folder "
            f"was moved or partly deleted"
        )
    return found


def _check_names(run_doc: Path, relatives: Sequence[str]) -> None:
    """Refuse a run document that gives two files of the dataset one path.

    The copy opens its destination exclusively and would stop part-way at the
    second; and a frame named ``run.json`` would be replaced by the generated
    one. Compared as a case-insensitive file system sees them (Windows, macOS):
    ``A1.tif`` and ``a1.TIF`` are one file there.
    """
    seen: dict[str, str] = {}
    for relative in [*relatives, _RUN_DOCUMENT, _README, _MANIFEST]:
        key = relative.casefold()
        if key in seen:
            raise DatasetError(
                f"{run_doc}: two files of the dataset would have the same path "
                f"({seen[key]} and {relative}); this is not a run document smc wrote"
            )
        seen[key] = relative


def _check_free_space(out: Path, need: int) -> None:
    """Refuse when the disk that will hold ``out`` has less free than ``need`` bytes.

    The disk is the one of the nearest folder that exists. A share that does
    not say how much is free is not a reason to refuse: unknown beats guessed,
    and a write that fills the disk is reported by the copy itself.
    """
    try:
        probe = out
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < need:
        raise DatasetError(
            f"{out}: the disk has {free / 1e6:,.1f} MB free and the dataset needs "
            f"{need / 1e6:,.1f} MB: free some space, or name a folder on another disk"
        )


def _check_path_lengths(out: Path, root: Path, relatives: Sequence[str]) -> None:
    """On Windows, refuse a destination whose path is ``MAX_PATH`` characters or more.

    Long paths are a setting of the lab PC, not something to assume. The
    generated files are written through a temporary file that is the longest
    path a write makes, so it is the one measured for them.
    """
    if not ON_WINDOWS:
        return
    longest = max(
        [len(str(root / relative)) for relative in relatives]
        + [
            len(str(temporary_path(root / name)))
            for name in (_RUN_DOCUMENT, _README, _MANIFEST)
        ]
    )
    if longest >= MAX_PATH:
        raise DatasetError(
            f"{out}: a file of the dataset would have a path of {longest} "
            f"characters, and Windows takes {MAX_PATH - 1} at most unless long "
            f"paths are enabled (not assumed): name a shorter folder"
        )


def export_dataset(
    run_doc: Path | str,
    out: Path | str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> DatasetManifest:
    """Write the scan of ``run_doc`` as a dataset in the new or empty folder ``out``.

    Everything that can be refused is refused before the first byte is written:
    ``out`` (a file, or a folder with something in it), the scan (``load_scan``
    refuses what is not a finished, real scan), missing frame files, a path
    that Windows would refuse, a disk with too little room. The scrubbed run
    document, the README and the manifest's fixed fields are built, and
    validated, before the first write too. Then the frames are copied in the
    scan's order, the cells file when the scan has cells, and ``run.json``,
    ``README.md`` and ``manifest.json`` last: a folder with no manifest is an
    export that did not finish (FM-106).

    Each copy hashes the **source's** bytes as it reads them, so the manifest
    is the truth of the source and a bad write is found by ``check`` (see
    ``_copy_and_hash``). Only the frames and the cells file are copied: the
    other stages' records stay in ``run.json`` (staleness reads them) and
    their files do not travel.

    Args:
        run_doc: The source run document's file.
        out: A folder that does not exist yet (its parents are made) or is empty.
        progress: Called as ``progress(done, total)`` after each file copied
            (the frames and the cells file; the three generated ones are not
            counted).

    Returns:
        The manifest, as written to ``out``.

    Raises:
        DatasetError: Refused; nothing was written (the missing parents of
            ``out`` are not made either).
        DatasetWriteError: An I/O error part-way; ``out`` holds part of the
            dataset and no ``manifest.json``, and nothing was deleted. When
            ``out`` itself could not be made nothing was written, and the
            message says so.
    """
    run_doc = Path(run_doc)
    out = Path(out)
    root = _check_out(out)
    scan = load_scan(run_doc)
    frames = _frame_sources(run_doc, scan)
    copies = [(relative, source) for relative, source, _ in frames]
    need = sum(size for _, _, size in frames) + _HEADROOM_BYTES
    cells_result = scan.cells_result
    if scan.cells is not None and cells_result is not None:
        try:
            cells_source = resolve_in_run(scan.run_dir, cells_result.file)
            need += cells_source.stat().st_size
        except (ValueError, OSError) as exc:
            raise DatasetError(
                f"{run_doc}: cells file {cells_result.file} cannot be read "
                f"({getattr(exc, 'strerror', None) or exc})"
            ) from exc
        copies.append((cells_result.file, cells_source))
    relatives = [relative for relative, _ in copies]
    _check_names(run_doc, relatives)
    _check_free_space(out, need)
    _check_path_lengths(out, root, relatives)

    scrubbed, pointers = scrub_document(scan.document)
    scan_record = scan.document.provider("scan")
    manifest = DatasetManifest(
        smc_version=__version__,
        exported_utc=datetime.now(timezone.utc),
        source=run_doc.name,
        profile=scan.profile,
        objective=scan.result.objective,
        pixel_size_um=scan.pixel_size_um,
        image_shape_px=scan.image_shape_px,
        wells=scan.wells,
        frames=len(scan.result.frames),
        cells=scan.cells is not None,
        cells_absent=scan.cells_absent,
        files={},
        scrubbed=pointers,
    )
    readme = readme_text(
        manifest,
        scanned_utc=None if scan_record is None else scan_record.started_utc,
        cells_file=None if cells_result is None else cells_result.file,
        classes=() if cells_result is None else cells_result.classes,
    )
    generated = [
        (_RUN_DOCUMENT, _json_bytes(scrubbed)),
        # A lone surrogate in an objective's name (FM-05) must not end the export.
        (_README, readme.encode("utf-8", errors="replace")),
    ]

    total = len(copies) + len(generated) + 1
    files: dict[str, DatasetFile] = {}
    where = "the folder itself"
    try:
        root.mkdir(parents=True, exist_ok=True)
        for relative, source in copies:
            where = relative
            size, digest = _copy_and_hash(source, root / relative)
            files[relative] = DatasetFile(bytes=size, sha256=digest)
            if progress is not None:
                progress(len(files), len(copies))
        for relative, data in generated:
            where = relative
            write_atomically(root / relative, data)
            files[relative] = DatasetFile(
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        final = manifest.model_copy(update={"files": files})
        where = _MANIFEST
        write_atomically(root / _MANIFEST, _json_bytes(final))
    except OSError as exc:
        if where == "the folder itself":
            # The first `mkdir` failed (a drive that is not there): no folder, no
            # file; "holds part of the dataset: remove it" would be false.
            raise DatasetWriteError(
                f"{out}: cannot be created ({exc.strerror or exc}); nothing was "
                f"written: name a folder on a disk that answers"
            ) from exc
        raise DatasetWriteError(
            f"{out}: stopped after {len(files)} of {total} files at {where}: "
            f"{exc.strerror or exc}; {out} holds part of the dataset and has no "
            f"manifest.json: remove it, then export again"
        ) from exc
    return final


@dataclass(frozen=True, slots=True)
class DatasetCheck:
    """What ``check_dataset`` found: the manifest, each file that does not match, and the scan.

    ``problems`` is one sentence per file, in the manifest's order, each
    beginning with the file's relative path; empty when every file matches.
    ``scan`` is the loaded scan when there is no problem, and ``None`` when
    there is one: the loader is not run on a folder that is known to be damaged.
    """

    manifest: DatasetManifest
    problems: tuple[str, ...]
    scan: Scan | None


def _read_manifest(directory: Path) -> tuple[Path, DatasetManifest]:
    """``directory/manifest.json``, read and validated, or a refusal that starts with its path.

    The manifest is read as ``utf-8-sig`` (a byte-order mark from Notepad,
    FM-80); its ``schema_version`` is judged on the raw JSON, before the model
    sees it, so that a newer smc's manifest is reported as newer and not as a
    list of fields this smc does not know; a manifest with no ``run.json`` in
    it is not one this smc wrote (FM-229).
    """
    path = directory / _MANIFEST
    try:
        data = retry_on_permission_error(path.read_bytes)
    except FileNotFoundError:
        raise DatasetError(
            f"{path}: no manifest.json: this is not a dataset, or an export that "
            f"did not finish (it writes manifest.json last)"
        ) from None
    except OSError as exc:
        raise DatasetError(f"{path}: cannot be read ({exc.strerror or exc})") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"{path}: is not UTF-8 text ({exc.reason} at byte {exc.start})"
        ) from None
    try:
        raw: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"{path}: is not valid JSON: {exc.msg} (line {exc.lineno}, column "
            f"{exc.colno})"
        ) from None
    if not isinstance(raw, dict):
        raise DatasetError(
            f"{path}: holds a JSON {type(raw).__name__}, not an object: this is "
            f"not a manifest"
        )
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise DatasetError(
            f"{path}: schema_version must be an integer of 1 or more, got "
            f"{version!r}: smc always writes one, so this is not a manifest it wrote"
        )
    if version > MANIFEST_SCHEMA:
        raise DatasetError(
            f"{path}: written by a newer smc (schema {version}); this smc reads "
            f"schema {MANIFEST_SCHEMA}: upgrade smc"
        )
    try:
        manifest = DatasetManifest.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "manifest"
        more = f" (and {exc.error_count() - 1} more)" if exc.error_count() > 1 else ""
        raise DatasetError(f"{path}: {where}: {first['msg']}{more}") from None
    if _RUN_DOCUMENT not in manifest.files:
        raise DatasetError(
            f"{path}: lists no {_RUN_DOCUMENT}: this is not a dataset this smc wrote"
        )
    return path, manifest


def check_dataset(
    directory: Path | str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> DatasetCheck:
    """Check a dataset folder against its manifest: every listed file's size and hash, then the scan.

    A copy that lost a byte on the way (a share, a zip, a USB stick) is found
    here and not by a wrong result a week later. Each listed file is read once
    (FM-39): its size is compared first and a file of another size is not read
    at all. The listed paths are judged before any file is opened: a manifest
    that names a file outside the folder (``../x``, an absolute path) is
    refused, so that ``check`` is never made to read a file of the PC by a
    manifest someone sent. Files of the folder that the manifest does not list
    are ignored: the collaborator's own outputs live beside the dataset.

    ``load_scan`` runs only when every file matched: it would read the damaged
    ones.

    Args:
        directory: The dataset folder.
        progress: Called as ``progress(done, total)`` after each file checked.

    Raises:
        DatasetError: ``directory`` is not a folder, has no readable manifest
            (an export that did not finish has none), the manifest lists a
            path outside the folder, or ``load_scan`` refuses the scan. A file
            that is missing or differs is a *problem* in the result, not an
            exception.
    """
    directory = Path(directory)
    try:
        is_folder = directory.is_dir()
    except OSError:
        is_folder = False
    if not is_folder:
        raise DatasetError(f"{directory}: is not a folder")
    path, manifest = _read_manifest(directory)
    listed: dict[str, Path] = {}
    for key in manifest.files:
        try:
            listed[key] = resolve_in_run(directory, key)
        except ValueError as exc:
            raise DatasetError(
                f"{path}: lists {key!r}, which is not a path in the dataset "
                f"folder: {exc}"
            ) from exc

    problems: list[str] = []
    total = len(listed)
    for done, (key, file) in enumerate(manifest.files.items(), start=1):
        problem = _problem_with(key, listed[key], file)
        if problem is not None:
            problems.append(problem)
        if progress is not None:
            progress(done, total)
    scan = None if problems else load_scan(directory / _RUN_DOCUMENT)
    return DatasetCheck(manifest=manifest, problems=tuple(problems), scan=scan)


def _problem_with(key: str, path: Path, listed: DatasetFile) -> str | None:
    """What is wrong with one listed file, or ``None``: missing, another size, other content, unreadable."""
    try:
        have = path.stat().st_size
    except FileNotFoundError:
        return f"{key}: missing"
    except OSError as exc:
        return f"{key}: cannot be read ({exc.strerror or exc})"
    if have != listed.bytes:
        return f"{key}: changed (size {have} instead of {listed.bytes})"
    try:
        digest = _sha256_of(path)
    except OSError as exc:
        return f"{key}: cannot be read ({exc.strerror or exc})"
    if digest != listed.sha256.lower():
        return f"{key}: changed (same size, other content)"
    return None
