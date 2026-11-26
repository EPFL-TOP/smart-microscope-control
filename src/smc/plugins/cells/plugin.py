"""``cells`` detects single and doublet cells on a scan's frames and counts each once (m3 §9.3).

The pure part (suppression, pixels to stage µm, the merge across fields) is
``algorithm.py`` and the model is ``smc.inference``'s: a name that a card gives
the weights, the classes and the pre-processing of (inference.md §6). This is
the loop around them, and the decisions in it that keep a long run from losing
or faking its work:

* **A well is the unit of work.** The cells of the wells done go to a partial
  file *before* the checkpoint names the well, so a crash between the two leaves
  a file that has more than the checkpoint says, never less. A resume keeps the
  cells of the wells the checkpoint names and drops the rest.
* **The model is part of the result.** Its identity, the weights' SHA-256 and a
  digest of the pre-processing the run applied, is in the checkpoint and in the
  result, and a resume with another identity is refused, naming the half that
  changed: cells of two models are not one result (FM-111). The result names the
  model by its card's name, never a path, since run documents are attached to
  issues (FM-50). The parameter ``model_path`` is the one place a path is
  recorded, as given, like every parameter: ``SMC_MODELS`` is not recorded
  anywhere. ``detector=blobs`` (#260) has no file: the SHA-256 of its identity
  string stands for it, and the result says that its counts are spots, not
  cells.
* **The model may run on a server** (``SMC_MODEL_SERVER``, inference.md §8): the
  name resolves through ``load_model`` either way, and a call that fails ends the
  stage ``failed`` with the server's diagnosis, never a frame with no cell and never
  a model quietly loaded here. The check of the server is reported at the start, the
  wells done stay in the checkpoint, and a resume goes on from them on either place
  and says so. Nothing recorded holds the server's address (FM-50).
* **A frame is checked before it is detected on.** The matrix of the scan is
  only right on the scan's own frame shape (FM-73), and a missing, corrupt or
  foreign frame is told by its file name with what to do.
* **Nothing is faked to look whole.** A scan with no frame, a label the
  detector does not declare, a frame that cannot be read: each ends the stage
  as ``failed`` with the checkpoint kept. A partial file that cannot be removed
  is only a WARNING: the result is whole.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import tifffile
from pydantic import BaseModel, Field, model_validator

from smc.inference import (
    DEFAULT_MODEL_NAME,
    ModelCard,
    ModelError,
    ModelPreprocessing,
    ResolvedModel,
    load_model,
    model_card,
)
from smc.inference.blobs import blob_model
from smc.inference.preprocessing import DEFAULT_PREPROCESSING
from smc.inference.served import ServedDetector
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.cells.algorithm import (
    merge_across_fields,
    suppress_in_frame,
    to_cells,
)
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
#: What the result says of a run of the blob detector: a count of spots is not a
#: count of cells, and a report that quotes it must say where it came from.
_BLOB_ASSUMED = (
    "cells were found by the blob detector (detector=blobs), which counts bright "
    "round spots for the synthetic plate of the demo-sample profile; on a real "
    "sample its counts are not cells",
)
#: What the model was trained with (``normalize_plane``'s defaults): the one pair of
#: percentiles a run of the blob detector may write out, since it has none of its own
#: and a run document recorded before #684 holds them for every run.
_TRAINED_LOW_PERCENTILE = DEFAULT_PREPROCESSING.percentiles[0]
_TRAINED_HIGH_PERCENTILE = DEFAULT_PREPROCESSING.percentiles[1]
#: A scan whose pixel size is further than this, relative to the checked one, from the
#: pixel size the model was measured at is not covered by that measurement.
_CHECKED_TOLERANCE = 0.05


class CellsParams(PluginParams):
    """Which detector, which weights, on which device, and how sure and how alike a detection must be."""

    detector: Literal["model", "blobs"] = Field(
        "model",
        description=(
            "model is the trained network. blobs counts bright round spots and is "
            "for the synthetic plate of the demo-sample profile only: on a real "
            "sample its counts are not cells."
        ),
    )
    model: str | None = Field(
        None,
        description=(
            "The model, by the name of its card in src/smc/inference/cards/ "
            f"(unset is {DEFAULT_MODEL_NAME}): its weights, classes and "
            "pre-processing. Refused with detector=blobs."
        ),
    )
    model_path: Path | None = Field(
        None,
        description=(
            "The weights, in place of the card's file in $SMC_MODELS (or in "
            "local/models of the checkout); its SHA-256 must be the card's."
        ),
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto",
        description=(
            "auto is cuda when there is one, else cpu; mps only when asked. With "
            "SMC_MODEL_SERVER set the model runs on the server, which chooses its "
            "own device: only auto is accepted."
        ),
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
    model_um_per_px: float | None = Field(
        None,
        gt=0.0,
        le=10.0,
        description=(
            "The pixel size the model sees, in µm: each frame is resized to it (the "
            "frame's pixel size over this one). Unset is the card's, which keeps "
            "the model's own resize (800 px) for cell-classes-v2. Refused with "
            "detector=blobs."
        ),
    )
    low_percentile: float | None = Field(
        None,
        ge=0.0,
        lt=50.0,
        description=(
            "The intensity at this percentile of a frame is black for the model. "
            "Unset is the card's (0.5 for cell-classes-v2, as it was trained). "
            "Refused with detector=blobs unless it is 0.5."
        ),
    )
    high_percentile: float | None = Field(
        None,
        gt=50.0,
        le=100.0,
        description=(
            "The intensity at this percentile of a frame is white for the model. "
            "Unset is the card's (99.5 for cell-classes-v2, as it was trained). "
            "Refused with detector=blobs unless it is 99.5."
        ),
    )
    checked_um_per_px: float | None = Field(
        None,
        gt=0.0,
        le=10.0,
        description=(
            "The frames' pixel size, in µm, at which this model and these parameters "
            "were measured against annotated frames (scripts/dev/cells_check.py). It "
            "changes nothing in the detection: the result says so when the scan's "
            "pixel size is not within 5 % of it, or when it is unset. Refused with "
            "detector=blobs."
        ),
    )

    @model_validator(mode="after")
    def _blobs_take_no_weights_and_no_gpu(self) -> CellsParams:
        # The blob detector is numpy: a path or a GPU asked for next to it would
        # be silently ignored, and the run would not be the one that was asked for.
        # The same goes for what only the model has: a resize, percentiles that
        # scale its input, and a pixel size it was checked at.
        if self.detector == "blobs":
            if self.model is not None:
                raise ValueError(
                    "model is for detector=model: the blob detector is not a named "
                    "model; remove model, or use detector=model"
                )
            if self.model_path is not None:
                raise ValueError(
                    "model_path is for detector=model: the blob detector has no "
                    "weights; remove model_path, or use detector=model"
                )
            if self.device not in ("auto", "cpu"):
                raise ValueError(
                    f"the blob detector runs on the cpu, not on {self.device}: "
                    "remove device, or use detector=model"
                )
            if self.model_um_per_px is not None:
                raise ValueError(
                    "model_um_per_px is for detector=model: the blob detector does "
                    "not resize the frame; remove model_um_per_px, or use "
                    "detector=model"
                )
            if self.checked_um_per_px is not None:
                raise ValueError(
                    "checked_um_per_px is for detector=model: the blob detector is "
                    "not a trained model that was checked; remove "
                    "checked_um_per_px, or use detector=model"
                )
            # A percentile written out as the trained one passes: a run document
            # recorded before #684 holds 0.5 and 99.5 for a blobs run, and a page
            # that re-validates it must not fail.
            if (
                self.low_percentile is not None
                and self.low_percentile != _TRAINED_LOW_PERCENTILE
            ):
                raise ValueError(
                    "low_percentile is for detector=model: the blob detector does "
                    "not scale its input by percentiles; remove low_percentile, or "
                    "use detector=model"
                )
            if (
                self.high_percentile is not None
                and self.high_percentile != _TRAINED_HIGH_PERCENTILE
            ):
                raise ValueError(
                    "high_percentile is for detector=model: the blob detector does "
                    "not scale its input by percentiles; remove high_percentile, "
                    "or use detector=model"
                )
        return self


class CellsCheckpoint(BaseModel):
    """What a resume needs: the files' token, the wells whose cells are in the partial file, the model's identity.

    ``identity`` is the weights' SHA-256 and the digest of the pre-processing,
    joined by ``+`` (``smc.inference.model_identity``), so that a resume can say
    which half changed. ``merged`` is the number of cells removed so far, so the
    result of a resumed run counts the wells it did not redo. A checkpoint saved
    before #684 has ``model_sha256`` and no ``identity``: it is refused as one
    that is not what ``cells`` expects. ``runs_on`` is where the model ran, as
    ``ResolvedModel.runs_on`` says it (``served``, ``local:cuda``), and has a default
    so that a checkpoint saved before #685 still loads: a resume does not refuse
    another place (a server that died, and the operator going local, is the case a
    resume is kept for), it says so in the result. When the places of two attempts
    differ the value lists both, joined by ``+``: the wells done are of both.
    """

    token: str
    wells_done: tuple[str, ...]
    identity: str
    merged: int = Field(ge=0)
    runs_on: str | None = None


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
    #: Never drives the stand, so ``smc run cells --offline`` runs it (algorithms §4).
    offline: ClassVar[bool] = True

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

        resolved = _resolve(params, scan)
        detector = resolved.detector
        if isinstance(detector, ServedDetector):
            _report_check(ctx, detector, len(scan.frames))

        index = scan.frame_index()
        token, cells, wells_done, merged, earlier = _resume(
            ctx, scan, resolved, detector.classes
        )
        partial = ctx.run_dir / "cells" / f"{token}.partial.json"
        done = sum(len(index[well]) for well in wells_done)
        resumed = done
        places = _places(earlier, resolved.runs_on)

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
                try:
                    detections = detector.detect(pixels)
                except ModelError as exc:
                    # A call that fails is a finding, never a frame with no cell: the
                    # stage ends `failed`, the checkpoint keeps the wells done, and
                    # `--resume` goes on from them (inference.md §8.7). The text is
                    # the server's diagnosis, which holds no address.
                    raise PluginRunError(_NAME, str(exc)) from exc
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
                    identity=resolved.identity,
                    merged=merged,
                    runs_on=places,
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
            model=resolved.name,
            model_sha256=resolved.weights_sha256,
            model_identity=resolved.identity,
            model_runs_on=resolved.runs_on,
            classes=detector.classes,
            device=detector.device,
            score_threshold=params.score_threshold,
            merge_radius_um=radius,
            frames=len(scan.frames),
            counts=counts,
            merged=merged,
            seconds_per_frame=sum(seconds) / len(seconds) if seconds else None,
            assumed=(
                *(
                    _BLOB_ASSUMED
                    if params.detector == "blobs"
                    else _unchecked(params, scan)
                ),
                *(
                    (
                        f"cells within {radius:g} µm seen by two fields are counted "
                        f"once: a chosen radius, not measured",
                    )
                    if radius > 0.0
                    else ()
                ),
                *_served_timings(resolved, seconds),
                *_other_place(earlier, resolved.runs_on, resumed, len(scan.frames)),
            ),
        )


def _resolve(params: CellsParams, scan: ScanResult) -> ResolvedModel:
    """The model the run uses: the blob detector, or a card's model with this run's rule.

    The frame's pixel size is ``scan.pixel_size_um``, the scan plan's, and not one
    derived from the matrix. ``load_model`` and ``model_card`` are called through
    this module's names, which is where a test replaces them.

    Raises:
        PluginRunError: The model name is unknown, its card or its weights are
            refused, or torch cannot be used: ``smc.inference``'s message, as it
            is, since it already says what to do.
    """
    if params.detector == "blobs":
        # No file to look for or to hash: the identity string stands for the
        # detector, so a resume across a change of its rules is refused like
        # a change of weights (FM-111). Neither torch nor the weights are touched.
        return blob_model()
    try:
        card = model_card(DEFAULT_MODEL_NAME if params.model is None else params.model)
        return load_model(
            card,
            preprocessing=_rule(card, params),
            frame_shape_px=scan.image_shape_px,
            frame_um_per_px=scan.pixel_size_um,
            device=params.device,
            score_threshold=params.score_threshold,
            weights=params.model_path,
        )
    except ModelError as exc:
        raise PluginRunError(_NAME, str(exc)) from exc


def _rule(card: ModelCard, params: CellsParams) -> ModelPreprocessing:
    """The card's pre-processing, with each parameter that is set in place of its own part.

    Unset is the card's: a stand that sets none of the three gets the model as it
    was trained. The card's bounds are the parameters', so the pair that comes out
    is always a valid one (``ModelPreprocessing`` checks it anyway).
    """
    low, high = card.preprocessing.percentiles
    size = card.preprocessing.model_um_per_px
    return ModelPreprocessing(
        percentiles=(
            low if params.low_percentile is None else params.low_percentile,
            high if params.high_percentile is None else params.high_percentile,
        ),
        model_um_per_px=size
        if params.model_um_per_px is None
        else params.model_um_per_px,
    )


def _unchecked(params: CellsParams, scan: ScanResult) -> tuple[str, ...]:
    """The line that says the model was not measured at this scan's pixel size, or none.

    The measurement of a model on annotated frames (``scripts/dev/cells_check.py``)
    holds for the pixel size of those frames. ``checked_um_per_px`` is the stand's
    record of it: unset, nothing was measured; set and further than 5 % from the
    scan's pixel size, the counts of this scan are at a pixel size nobody measured.
    The test is written so that a number that is not one says it (FM-104).
    """
    checked = params.checked_um_per_px
    if checked is None:
        return (
            "the model's detections were not measured on frames of this stand: "
            "checked_um_per_px is not set (docs/plugins/cells.md)",
        )
    pixel = scan.pixel_size_um
    if not abs(pixel / checked - 1.0) <= _CHECKED_TOLERANCE:
        return (
            f"the frames are at {pixel:.4g} µm per pixel and the model was checked "
            f"at {checked:.4g} µm per pixel: its counts at this pixel size are not "
            f"measured",
        )
    return ()


def _identity_changed(was: str, now: str) -> str:
    """What a resume says when the model is not the one that made the run: which half changed.

    An identity is the weights' SHA-256 and the pre-processing's digest joined by
    ``+``, so the weights are told apart from the rule the frames were shown
    through: another file is one cure (put the weights back), another rule the
    other (set the parameters as they were).
    """
    if was.partition("+")[0] != now.partition("+")[0]:
        return (
            "the weights changed since the run being resumed; run cells "
            "without --resume"
        )
    return (
        "the pre-processing changed since the run being resumed; run cells "
        "without --resume"
    )


def _resume(
    ctx: RunContext,
    scan: ScanResult,
    resolved: ResolvedModel,
    classes: tuple[str, ...],
) -> tuple[str, list[Cell], list[str], int, str | None]:
    """The token, the cells of the wells done, those wells, their ``merged`` and where they were detected; or a fresh start.

    Without a checkpoint the token is new and nothing is done. With one, the
    model's identity must be the one that made it, and its token is kept so that the
    final file is the one the partial file leads to. The place is **not** compared:
    a server that died and an operator who goes local is the case a resume is for,
    so the place the checkpoint names is returned (``None`` for none, or when no
    well is kept) and the result says so. A missing or invalid
    partial file means no well is done: every well is detected again under the
    same token, which is slower and never wrong (``_redo_every_well`` resets the
    stored checkpoint first). A cell of the partial file whose well is not in
    ``wells_done`` came from a crash between the file and the checkpoint, and is
    dropped. What a valid partial file cannot show is a well that has no cell:
    a copy of the run folder made mid-run, with an older partial file than its
    checkpoint, would be taken at its word (a count of cells in the checkpoint
    would tell; it is a follow-up, the checkpoint's fields being fixed).
    """
    identity = resolved.identity
    checkpoint = ctx.checkpoint(CellsCheckpoint)
    if checkpoint is None:
        return new_token(), [], [], 0, None
    if checkpoint.identity != identity:
        raise PluginRunError(_NAME, _identity_changed(checkpoint.identity, identity))
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
        return _redo_every_well(ctx, checkpoint.token, resolved)
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
            return _redo_every_well(ctx, checkpoint.token, resolved)
    return checkpoint.token, kept, done, checkpoint.merged, checkpoint.runs_on


def _redo_every_well(
    ctx: RunContext, token: str, resolved: ResolvedModel
) -> tuple[str, list[Cell], list[str], int, str | None]:
    """Nothing is done, and the stored checkpoint says so before the first well is redone.

    The redo rewrites the partial file well by well. A stored checkpoint that
    still named the wells of the lost file would, if the run ended before its
    first new checkpoint, make the next resume trust a partial file that holds
    fewer wells than that, skip the rest, and record a ``done`` result whose
    counts for them are zero (FM-111). No well is kept, so no earlier place is
    reported.
    """
    ctx.save_checkpoint(
        CellsCheckpoint(
            token=token,
            wells_done=(),
            identity=resolved.identity,
            merged=0,
            runs_on=resolved.runs_on,
        )
    )
    return token, [], [], 0, None


def _report_check(ctx: RunContext, detector: ServedDetector, frames: int) -> None:
    """Say at the start of the stage what the check of the model server found.

    On the progress line and on the plugin's logger (WARNING when the check is not
    ok). It does not refuse the stage: a server that is not there ends it ``failed``
    at the first frame with the call's own diagnosis, and the operator has the
    reason from the first second, not after a frame that was read for nothing.
    """
    check = detector.check
    ctx.progress(0, frames, check.message)
    (ctx.logger.info if check.ok else ctx.logger.warning)("%s", check.message)


def _places(earlier: str | None, now: str) -> str:
    """Where the wells in the checkpoint were detected: the earlier places and this one.

    Equal places are one; places that differ are listed (``served+local:cpu``), so that
    a third attempt on the second place still knows that the first well was not detected
    there: a checkpoint that said only ``local:cpu`` would let it say nothing of it.
    """
    if earlier is None or earlier == now:
        return now
    if now in earlier.split("+"):
        return earlier
    return f"{earlier}+{now}"


def _other_place(
    earlier: str | None, now: str, resumed: int, frames: int
) -> tuple[str, ...]:
    """The line that says a resume detected some frames elsewhere than the rest, or none."""
    if earlier is None or earlier == now or resumed == 0:
        return ()
    return (
        f"{resumed} of {frames} frames were detected by an earlier attempt that ran "
        f"on {earlier}; the others ran on {now}",
    )


def _served_timings(
    resolved: ResolvedModel, seconds: Sequence[float]
) -> tuple[str, ...]:
    """What a served run took, with the network in it, or nothing for a local run.

    The first call carries the server's start-up work (kernels, the first allocation),
    so it is told apart from the mean of the next; neither is a measurement of the
    server alone, since the network is in both. Only the frames timed in *this*
    attempt count.
    """
    if resolved.runs_on != "served" or not seconds:
        return ()
    first, rest = seconds[0], seconds[1:]
    then = (
        f" and the next {len(rest)} frame{'s' if len(rest) > 1 else ''} "
        f"{sum(rest) / len(rest):.2f} s on average"
        if rest
        else ""
    )
    return (
        f"the model ran on the server: the first frame took {first:.2f} s{then}, "
        f"network included (the first can carry the server's start-up work); the "
        f"server's device was {resolved.detector.device}",
    )


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
