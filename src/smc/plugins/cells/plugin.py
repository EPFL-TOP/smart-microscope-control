"""``cells`` detects single and doublet cells on a scan's frames and counts each once (m3 §9.3).

The pure part (suppression, pixels to stage µm, the merge across fields) is
``algorithm.py`` and the model is ``detector.py``. This is the loop around them,
and the decisions in it that keep a long run from losing or faking its work:

* **A well is the unit of work.** The cells of the wells done go to a partial
  file *before* the checkpoint names the well, so a crash between the two leaves
  a file that has more than the checkpoint says, never less. A resume keeps the
  cells of the wells the checkpoint names and drops the rest.
* **The weights are part of the result.** Their SHA-256 is in the checkpoint
  and in the result, and a resume with other weights is refused: cells of two
  models are not one result (FM-111). The result names the weights' file, never
  its path, since run documents are attached to issues (FM-50). The parameter
  ``model_path`` is the one place a path is recorded, as given, like every
  parameter: ``SMC_MODELS`` is not recorded anywhere.
* **A frame is checked before it is detected on.** The matrix of the scan is
  only right on the scan's own frame shape (FM-73), and a missing, corrupt or
  foreign frame is told by its file name with what to do.
* **Nothing is faked to look whole.** A scan with no frame, a label the
  detector does not declare, a frame that cannot be read: each ends the stage
  as ``failed`` with the checkpoint kept. A partial file that cannot be removed
  is only a WARNING: the result is whole.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import tifffile
from pydantic import BaseModel, Field

from smc.plugins.api import PluginParams, RunContext
from smc.plugins.cells.algorithm import (
    merge_across_fields,
    suppress_in_frame,
    to_cells,
)
from smc.plugins.cells.detector import default_model_path, load_detector
from smc.plugins.errors import PluginRunError
from smc.plugins.slots import (
    Cell,
    CellsFile,
    CellsResult,
    ScanFrame,
    ScanResult,
    load_frame,
)
from smc.run.files import (
    AtomicWriteError,
    new_token,
    relative_to_run,
    retry_on_permission_error,
    write_atomically,
)

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["CellsCheckpoint", "CellsParams", "CellsPlugin"]

_NAME = "cells"
#: The weights are hashed in pieces of this size: 166 MB are not read at once.
_HASH_CHUNK_BYTES = 1 << 20


class CellsParams(PluginParams):
    """Which weights, on which device, and how sure and how alike a detection must be."""

    model_path: Path | None = Field(
        None,
        description=(
            "The weights; default local/models/cell_classes_model-cat-v2.pth of "
            "the checkout, or that file in $SMC_MODELS."
        ),
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto",
        description="auto is cuda when there is one, else cpu; mps only when asked.",
    )
    score_threshold: float = Field(
        0.5, ge=0.05, le=1.0, description="Detections below this score are dropped."
    )
    max_overlap: float = Field(
        0.7,
        gt=0.0,
        le=1.0,
        description=(
            "Two boxes of one frame sharing more than this fraction of the "
            "smaller one are one object; the more confident label stays."
        ),
    )
    merge_radius_um: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Cells of two fields within this distance are one cell; 0 merges nothing."
        ),
    )


class CellsCheckpoint(BaseModel):
    """What a resume needs: the files' token, the wells whose cells are in the partial file, the weights.

    ``merged`` is the number of cells removed so far, so the result of a
    resumed run counts the wells it did not redo.
    """

    token: str
    wells_done: tuple[str, ...]
    model_sha256: str
    merged: int = Field(ge=0)


class CellsPlugin:
    """Detects cells on every frame of a scan, in stage µm, each object counted once."""

    name = _NAME
    summary = (
        "Detect single and doublet cells on a scan's frames, in stage µm, each "
        "counted once."
    )
    Params = CellsParams
    Result = CellsResult
    requires: tuple[type, ...] = ()
    provides = "cells"
    reads: tuple[str, ...] = ("scan",)
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: CellsParams,
        ctx: RunContext,
    ) -> CellsResult:
        """Detect well by well, checkpointing after each, and write the cells file.

        The stand is not used: the frames are in the run folder. A dry run gets
        no special case, since a dry-run scan has no frame and is refused below.

        Raises:
            PluginRunError: The scan has no frame; the model, a frame or a file
                cannot be used; the weights changed since the run being
                resumed; the detector returned a label it does not declare.
            StopRequested: The operator stopped it; the checkpoint names the
                wells whose cells are saved.
        """
        scan = ctx.upstream("scan", ScanResult)
        if not scan.frames:
            raise PluginRunError(
                _NAME,
                "the scan has no frame (it was a dry run, or every field "
                "failed); run scan again without --dry-run",
            )

        weights = (
            params.model_path if params.model_path is not None else default_model_path()
        )
        detector = load_detector(
            weights, device=params.device, score_threshold=params.score_threshold
        )
        sha256 = _sha256(weights)

        index = scan.frame_index()
        token, cells, wells_done, merged = _resume(ctx, scan, sha256, detector.classes)
        partial = ctx.run_dir / "cells" / f"{token}.partial.json"
        done = sum(len(index[well]) for well in wells_done)

        seconds: list[float] = []
        for well, positions in index.items():
            if well in wells_done:
                continue
            well_cells: list[Cell] = []
            for position in positions:
                ctx.raise_if_stopped()
                frame = scan.frames[position]
                pixels = _read_frame(ctx.run_dir, frame, scan.image_shape_px)
                started = time.perf_counter()
                detections = detector.detect(pixels)
                seconds.append(time.perf_counter() - started)
                unknown = {d.label for d in detections} - set(detector.classes)
                if unknown:
                    raise PluginRunError(
                        _NAME,
                        f"the detector returned the label {sorted(unknown)[0]!r} for "
                        f"{frame.file}, which is not one of its classes "
                        f"{list(detector.classes)}",
                    )
                well_cells += to_cells(
                    suppress_in_frame(detections, max_overlap=params.max_overlap),
                    frame,
                    position,
                    matrix=scan.matrix_um_per_px,
                    image_shape_px=scan.image_shape_px,
                )
                done += 1
                ctx.progress(done, len(scan.frames), f"{well} {frame.name}")
            kept, removed = merge_across_fields(
                well_cells,
                image_shape_px=scan.image_shape_px,
                radius_um=params.merge_radius_um,
            )
            cells += kept
            merged += removed
            wells_done.append(well)
            # The file before the checkpoint: a crash between the two leaves
            # cells that no checkpoint names, which a resume drops.
            _write_cells(ctx.run_dir, partial, cells)
            ctx.save_checkpoint(
                CellsCheckpoint(
                    token=token,
                    wells_done=tuple(wells_done),
                    model_sha256=sha256,
                    merged=merged,
                )
            )

        final = ctx.run_dir / "cells" / f"{token}.json"
        _write_cells(ctx.run_dir, final, cells)
        _remove_partial(ctx, partial)

        counts = {well: dict.fromkeys(detector.classes, 0) for well in index}
        for cell in cells:
            counts[cell.well][cell.label] += 1
        radius = params.merge_radius_um
        return CellsResult(
            file=relative_to_run(ctx.run_dir, final),
            model=weights.name,
            model_sha256=sha256,
            classes=detector.classes,
            device=detector.device,
            score_threshold=params.score_threshold,
            merge_radius_um=radius,
            frames=len(scan.frames),
            counts=counts,
            merged=merged,
            seconds_per_frame=sum(seconds) / len(seconds) if seconds else None,
            assumed=(
                (
                    f"cells within {radius:g} µm seen by two fields are counted "
                    f"once: a chosen radius, not measured",
                )
                if radius > 0.0
                else ()
            ),
        )


def _sha256(path: Path) -> str:
    """The SHA-256 of the weights, read in pieces: which model made these cells.

    Raises:
        PluginRunError: The file cannot be read, which the detector could a
            moment ago: it was replaced or locked meanwhile.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise PluginRunError(
            _NAME,
            f"{path.name} cannot be read to hash it ({exc.strerror or exc}); "
            f"is another program using it?",
        ) from exc
    return digest.hexdigest()


def _resume(
    ctx: RunContext, scan: ScanResult, sha256: str, classes: tuple[str, ...]
) -> tuple[str, list[Cell], list[str], int]:
    """The token, the cells of the wells done, those wells and their ``merged``; or a fresh start.

    Without a checkpoint the token is new and nothing is done. With one, the
    weights must be the ones that made it, and its token is kept so that the
    final file is the one the partial file leads to. A missing or invalid
    partial file means no well is done: every well is detected again under the
    same token, which is slower and never wrong (``_redo_every_well`` resets the
    stored checkpoint first). A cell of the partial file whose well is not in
    ``wells_done`` came from a crash between the file and the checkpoint, and is
    dropped. What a valid partial file cannot show is a well that has no cell:
    a copy of the run folder made mid-run, with an older partial file than its
    checkpoint, would be taken at its word (a count of cells in the checkpoint
    would tell; it is a follow-up, the checkpoint's fields being fixed).
    """
    checkpoint = ctx.checkpoint(CellsCheckpoint)
    if checkpoint is None:
        return new_token(), [], [], 0
    if checkpoint.model_sha256 != sha256:
        raise PluginRunError(
            _NAME,
            "the weights changed since the run being resumed; run cells "
            "without --resume",
        )
    partial = ctx.run_dir / "cells" / f"{checkpoint.token}.partial.json"
    where = relative_to_run(ctx.run_dir, partial)
    try:
        loaded = CellsFile.model_validate_json(
            retry_on_permission_error(partial.read_bytes)
        )
    except (OSError, ValueError) as exc:
        ctx.logger.warning(
            "%s cannot be used (%s): every well is detected again",
            where,
            _why(exc),
        )
        return _redo_every_well(ctx, checkpoint.token, sha256)
    index = scan.frame_index()
    done = [well for well in checkpoint.wells_done if well in index]
    kept = [cell for cell in loaded.cells if cell.well in done]
    for cell in kept:
        if cell.label not in classes:
            ctx.logger.warning(
                "%s holds the label %r, which the detector does not declare: "
                "every well is detected again",
                where,
                cell.label,
            )
            return _redo_every_well(ctx, checkpoint.token, sha256)
    return checkpoint.token, kept, done, checkpoint.merged


def _redo_every_well(
    ctx: RunContext, token: str, sha256: str
) -> tuple[str, list[Cell], list[str], int]:
    """Nothing is done, and the stored checkpoint says so before the first well is redone.

    The redo rewrites the partial file well by well. A stored checkpoint that
    still named the wells of the lost file would, if the run ended before its
    first new checkpoint, make the next resume trust a partial file that holds
    fewer wells than that, skip the rest, and record a ``done`` result whose
    counts for them are zero (FM-111).
    """
    ctx.save_checkpoint(
        CellsCheckpoint(token=token, wells_done=(), model_sha256=sha256, merged=0)
    )
    return token, [], [], 0


def _why(exc: Exception) -> str:
    """A short reason for a log line: the OS's words, or the first line of a validation error."""
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return " ".join(str(exc).split())[:200]


def _read_frame(
    run_dir: Path, frame: ScanFrame, image_shape_px: tuple[int, int]
) -> npt.NDArray[Any]:
    """The frame's pixels, once it is known that the scan's matrix is right for them.

    Every failure names ``frame.file`` and says to run ``scan`` again: the frame
    is the scan's product, and nothing here can make a better one.
    """
    again = "run scan again"
    try:
        pixels = load_frame(run_dir, frame)
    except FileNotFoundError as exc:
        raise PluginRunError(
            _NAME, f"frame file {frame.file} is missing from the run folder; {again}"
        ) from exc
    except (ValueError, OSError, tifffile.TiffFileError) as exc:
        # A path out of the folder is a ValueError, a locked file an OSError. A
        # corrupt TIFF is `TiffFileError`, a ValueError only in the newer tifffile:
        # the 2025.5.10 that Python 3.10 installs derives it from Exception.
        raise PluginRunError(
            _NAME, f"frame file {frame.file} cannot be read ({_why(exc)}); {again}"
        ) from exc
    if pixels.ndim != 2 or tuple(pixels.shape) != tuple(image_shape_px):
        raise PluginRunError(
            _NAME,
            f"frame file {frame.file} has shape {tuple(pixels.shape)}, the scan's "
            f"frames are {tuple(image_shape_px)}; {again}",
        )
    if pixels.dtype.kind == "f" and not np.isfinite(pixels).all():
        raise PluginRunError(
            _NAME, f"frame file {frame.file} has a value that is not finite; {again}"
        )
    return pixels


def _write_cells(run_dir: Path, path: Path, cells: list[Cell]) -> None:
    """Write ``cells`` as a ``CellsFile``: ASCII JSON, all or nothing (FM-38).

    ASCII with escapes, as the run document: the file is read back on Windows,
    and ``nan`` cannot be written.

    Raises:
        PluginRunError: The file cannot be written; it is as it was, and the
            checkpoint still names the wells done.
    """
    text = json.dumps(
        CellsFile(cells=tuple(cells)).model_dump(mode="json"),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        write_atomically(path, text.encode("ascii"))
    except AtomicWriteError as exc:
        where = relative_to_run(run_dir, path)
        if exc.kept is not None:
            # Only the rename failed, which on Windows is a program holding the
            # file: the whole new content waits in a hidden file (FM-38).
            advice = (
                f"the whole new file is kept as {relative_to_run(run_dir, exc.kept)}; "
                f"close any program that holds {where}, then run with --resume"
            )
        else:
            advice = "free some space or check the run folder, then run with --resume"
        raise PluginRunError(
            _NAME, f"cannot write {where} ({exc.strerror or exc}); {advice}"
        ) from exc


def _remove_partial(ctx: RunContext, partial: Path) -> None:
    """Remove the partial file; a failure is a WARNING, because the result is whole."""
    try:
        retry_on_permission_error(lambda: partial.unlink(missing_ok=True))
    except OSError as exc:
        ctx.logger.warning(
            "%s could not be removed (%s); it is of no use and can be deleted",
            relative_to_run(ctx.run_dir, partial),
            _why(exc),
        )
