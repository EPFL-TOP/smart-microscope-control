"""The cell detector: what the stage asks of one, and the trained model behind it (m3 §9.1).

``Detector`` is a protocol and there is no discovery in M3: one detector,
``TorchDetector``, and the stubs of the tests. The pure algorithm
(``algorithm.py``) only sees ``Detection``, so it runs on synthetic data with no
model.

Boxes are ``(row0, col0, row1, col1)`` in the pixels of the frame, as the cells
file stores them. torchvision emits ``[x0, y0, x1, y1]``; the swap is made once,
in ``TorchDetector.detect``, so nothing downstream thinks about it.

torch and torchvision are the ``detect`` extra and are imported **inside**
``TorchDetector.__init__`` only: this module, and the plugin that uses it,
import on a PC that never installed them, so ``smc run --list`` shows ``cells``
everywhere and a missing torch is an instruction, not a crash.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from skimage import measure

import smc
from smc.plugins.errors import PluginRunError

__all__ = [
    "BLOB_DETECTOR",
    "BLOB_DETECTOR_IDENTITY",
    "DEFAULT_MODEL",
    "BlobDetector",
    "Detection",
    "Detector",
    "TorchDetector",
    "default_model_path",
    "detections_from_output",
    "load_detector",
    "normalize_plane",
]

#: The trained single / doublet / debris model, trained by ``nikon-control``. It is
#: 166 MB and not in git: it lives in ``local/models`` (ignored) or in ``$SMC_MODELS``.
DEFAULT_MODEL = "cell_classes_model-cat-v2.pth"

_BACKGROUND = "__background__"
_PREDICTOR_KEY = "roi_heads.box_predictor.cls_score.weight"
_PLUGIN = "cells"

log = logging.getLogger(f"smc.plugins.{_PLUGIN}")


@dataclass(frozen=True, slots=True)
class Detection:
    """One box the detector found, in the pixels of its frame, with its label and score."""

    row0_px: float
    col0_px: float
    row1_px: float
    col1_px: float
    label: str
    score: float


@runtime_checkable
class Detector(Protocol):
    """What the cells stage asks of a detector: its labels, its device and ``detect``.

    ``classes`` are the labels it can return, background excluded; ``device`` is
    the one it runs on, resolved (never ``auto``), so the result can say what
    produced it.
    """

    classes: tuple[str, ...]
    device: str

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The cells of one 2-D frame, in no particular order."""
        ...


def normalize_plane(
    plane: npt.NDArray[Any], percentiles: tuple[float, float] = (0.5, 99.5)
) -> npt.NDArray[np.float32]:
    """A 2-D frame as float32 in [0, 1], clipped at two percentiles.

    The model was trained on percentile-normalised planes. Percentiles and not
    the minimum and maximum, so that a hot pixel or the bright rim of a well does
    not squash the cells into a few grey levels. A constant frame has no range to
    scale and comes out all 0 (the epsilon keeps the division finite).
    """
    values = np.asarray(plane, dtype=np.float32)
    low, high = np.percentile(values, percentiles)
    scaled = np.clip((values - low) / (high - low + 1e-6), 0.0, 1.0)
    return np.asarray(scaled, dtype=np.float32)


def detections_from_output(
    boxes: npt.NDArray[Any],
    scores: npt.NDArray[Any],
    label_ids: npt.NDArray[Any],
    *,
    labels: Sequence[str],
    score_threshold: float,
) -> list[Detection]:
    """The model's output as ``Detection`` s: the one place columns and rows change places.

    torchvision's box is ``[x0, y0, x1, y1]``, columns first; a ``Detection`` is
    ``(row0, col0, row1, col1)``, as the cells file stores it. A box swapped here
    is invisible on a square frame and wrong on a camera's (FM-73), so this is
    numpy only and tested without torch. ``labels`` is the checkpoint's full
    class list, background first: the model's label id indexes it directly.
    Scores below ``score_threshold`` are dropped; ``score_threshold`` itself stays.
    """
    return [
        Detection(
            float(y0),
            float(x0),
            float(y1),
            float(x1),
            labels[int(label_id)],
            float(score),
        )
        for (x0, y0, x1, y1), score, label_id in zip(
            boxes, scores, label_ids, strict=True
        )
        if score >= score_threshold
    ]


def default_model_path() -> Path:
    """Where the trained weights are looked for: ``$SMC_MODELS`` or the checkout's ``local/models``.

    An empty or blank ``SMC_MODELS`` is no setting at all (FM-78): ``Path("")`` is
    the working directory, and the weights would be looked for wherever the
    command was typed. The checkout is the editable install's, which is how
    every PC runs smc.
    """
    folder = os.environ.get("SMC_MODELS", "").strip()
    if folder:
        return Path(folder) / DEFAULT_MODEL
    return Path(smc.__file__).resolve().parents[2] / "local" / "models" / DEFAULT_MODEL


class TorchDetector:
    """The trained Faster R-CNN (``fasterrcnn_resnet50_fpn``) behind the ``Detector`` protocol.

    The checkpoint is read with ``weights_only=True``, which cannot run code, and
    must be the training's own: a ``model_state_dict`` and the ``classes`` it was
    trained with, background first. Everything that can be wrong with a file is
    checked before a model is built (a wrong file is told in a line, not as a
    shape mismatch of 40 tensors), and the weights are loaded with
    ``strict=True``. A device that was asked for and is not there is refused and
    never replaced: a scan that silently ran on the CPU is a scan nobody planned.

    Args:
        weights: The checkpoint file.
        device: ``auto`` (``cuda`` when available, else ``cpu``; never ``mps``
            unasked, some detection ops are not implemented there), ``cpu``,
            ``cuda`` or ``mps``.
        score_threshold: Detections below it are dropped.

    Raises:
        ImportError: torch or torchvision is not installed, or one of their
            compiled modules does not import.
        OSError, RuntimeError: torch is installed and does not load (a DLL on
            Windows; a torchvision of another version than torch).
        ValueError: The file is not a usable checkpoint, or the device is not
            available. The message names the file.
    """

    classes: tuple[str, ...]
    device: str

    def __init__(self, weights: Path, *, device: str, score_threshold: float) -> None:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        name = weights.name
        try:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            # Any reader error, a locked file (PermissionError) included, is this
            # file's problem: the caller catches OSError for torch's own imports.
            raise ValueError(
                f"{name} cannot be read as a checkpoint ({type(exc).__name__}: {exc})"
            ) from exc
        state = (
            checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        )
        if not isinstance(state, dict):
            raise ValueError(
                f"{name} is not a training checkpoint: it has no model_state_dict"
            )
        names = checkpoint.get("classes")
        if (
            not isinstance(names, list | tuple)
            or len(names) < 2
            or not all(isinstance(n, str) for n in names)
        ):
            raise ValueError(
                f"{name} has no classes: it needs the list of at least two "
                f"names it was trained with"
            )
        if names[0] != _BACKGROUND:
            raise ValueError(
                f"{name} lists {names[0]!r} as its first class, not {_BACKGROUND!r}"
            )
        # `shape` of nothing, of a list or of a number is `None`, and a 0-d tensor's
        # is empty: none of them is a predictor with one row per class.
        shape = getattr(state.get(_PREDICTOR_KEY), "shape", None)
        if not shape:
            raise ValueError(
                f"{name} has no usable {_PREDICTOR_KEY}: the box predictor's "
                f"weight is missing or is not a tensor with a row per class"
            )
        if shape[0] != len(names):
            raise ValueError(
                f"{name} has {shape[0]} outputs for {len(names)} classes: "
                f"it is not the model its classes name"
            )

        resolved = _resolve_device(torch, device)
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(names)
        )
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} does not fit the Faster R-CNN architecture ({exc}); "
                f"is it the right checkpoint?"
            ) from exc
        try:
            model.eval().to(resolved)
        except RuntimeError as exc:
            # `auto` picked cuda and the card cannot run this model (no kernel image
            # for its architecture, out of memory): say what to do, as for a device
            # that is not there.
            advice = "" if resolved == "cpu" else "; use device=cpu"
            raise ValueError(
                f"{name} could not be moved to {resolved} ({exc}){advice}"
            ) from exc

        self.score_threshold = score_threshold
        self.classes = tuple(names[1:])
        self.device = resolved
        self._labels = tuple(names)
        # torchvision keeps the best `detections_per_img` boxes of a frame (100 by
        # default, as in `nikon-control`) and says nothing about the rest.
        self._limit: int | None = getattr(
            getattr(model, "roi_heads", None), "detections_per_img", None
        )
        self._model = model
        self._torch = torch

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The cells of one frame above the score threshold, labelled from the full class list.

        One frame per forward pass, under ``no_grad``, with the model's own
        resize: batching frames is not decided here. The plane is repeated to
        three channels, which is what the backbone expects.

        Raises:
            ValueError: ``frame`` is not 2-D (a colour frame is not a brightfield one).
        """
        array = np.asarray(frame)
        if array.ndim != 2:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        torch = self._torch
        plane = torch.from_numpy(normalize_plane(array))
        tensor = plane[None].repeat(3, 1, 1).to(self.device)
        with torch.no_grad():
            out = self._model([tensor])[0]
        boxes = out["boxes"].cpu().numpy()
        if self._limit is not None and len(boxes) >= self._limit:
            log.warning(
                "the model returned %d boxes for a frame, its limit per frame: a "
                "denser frame is undercounted",
                len(boxes),
            )
        return detections_from_output(
            boxes,
            out["scores"].cpu().numpy(),
            out["labels"].cpu().numpy(),
            labels=self._labels,
            score_threshold=self.score_threshold,
        )


#: ``CellsResult.model`` when the blob detector ran: there is no file to name.
BLOB_DETECTOR = "blobs"
#: What the blob detector is, rule by rule. Its SHA-256 is ``CellsResult.model_sha256``
#: and the resume's check that two runs found cells the same way (FM-111): a change
#: of a rule below changes the version in this string.
BLOB_DETECTOR_IDENTITY = (
    "smc BlobDetector 1: p75 + 8 sigma, 9 px to 5 % of the frame, "
    "axis ratio >= 0.3, single from 0.75"
)
# scikit-image's `label` and `regionprops` are unannotated, which `mypy --strict`
# refuses at the call (`no-untyped-call`). Bound once, with what they return
# (a label image, and region objects that have the properties used below).
_label_image: Callable[..., npt.NDArray[np.int_]] = measure.label
_region_properties: Callable[..., Sequence[Any]] = measure.regionprops
_BLOB_SIGMAS = 8.0
_BLOB_MIN_AREA_PX = 9
_BLOB_MAX_AREA_FRACTION = 0.05
_BLOB_MIN_AXIS_RATIO = 0.3
_BLOB_SINGLE_AXIS_RATIO = 0.75


class BlobDetector:
    """Bright round spots as ``single`` and ``doublet``: the detector for the synthetic plate.

    The trained model does not see the cells of the demo-sample stand (at most 2
    ``single`` among 14 objects, m1 §8), so the workflow could not be followed on
    images past the scan. This finds what that sample draws and nothing else: it
    knows nothing of cells, and the result says so.

    The background is the frame's 75th percentile, so that the cells, a few
    percent of the frame, do not lift it. The noise is the spread of the
    differences of neighbouring pixels along a row, not of the pixels around the
    median: a well wall is a step, which a spread around the median counts as
    noise (a frame centred on a wall gave five false cells at 1 µm per pixel).
    A component is a cell between 9 pixels and 5 % of the frame: a larger one is
    a sliver of well seen past a wall, and at 2 % the doublets vanished at
    0.1625 µm per pixel. A strip is not a cell. A round component is a
    ``single``, an elongated one a ``doublet``.

    ``score`` is always 1.0: a threshold on it would mean nothing here.
    """

    classes: tuple[str, ...] = ("single", "doublet")
    device: str = "cpu"

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The spots of one frame, in no particular order; none on a flat or noiseless one.

        Raises:
            ValueError: ``frame`` is not 2-D.
        """
        array = np.asarray(frame)
        if array.ndim != 2:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        values = array.astype(np.float64)
        if values.shape[0] < 1 or values.shape[1] < 2:
            return []
        sigma = (
            1.4826 * float(np.median(np.abs(np.diff(values, axis=1)))) / math.sqrt(2)
        )
        if not (math.isfinite(sigma) and sigma > 0.0):
            return []
        threshold = float(np.percentile(values, 75)) + _BLOB_SIGMAS * sigma
        largest_px = _BLOB_MAX_AREA_FRACTION * values.size
        found: list[Detection] = []
        for region in _region_properties(_label_image(values > threshold)):
            major = float(region.axis_major_length)
            if (
                region.area < _BLOB_MIN_AREA_PX
                or region.area > largest_px
                or major <= 0
            ):
                continue
            ratio = float(region.axis_minor_length) / major
            if ratio < _BLOB_MIN_AXIS_RATIO:
                continue
            row0, col0, row1, col1 = region.bbox
            found.append(
                Detection(
                    float(row0),
                    float(col0),
                    float(row1),
                    float(col1),
                    "single" if ratio >= _BLOB_SINGLE_AXIS_RATIO else "doublet",
                    1.0,
                )
            )
        return found


def _resolve_device(torch: Any, requested: str) -> str:
    """The device to run on; one that was asked for and is not there is a ``ValueError``."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if requested == "mps" and mps is not None and mps.is_available():
        return "mps"
    if requested in ("cuda", "mps"):
        raise ValueError(
            f"device {requested} is not available on this PC; use device=auto or cpu"
        )
    raise ValueError(f"device {requested!r} is not one of auto, cpu, cuda, mps")


def load_detector(
    model_path: Path | None, *, device: str, score_threshold: float
) -> Detector:
    """The trained detector, or a ``PluginRunError`` that says what to do about it.

    The file is looked for first, so a PC with neither the weights nor torch is
    told about the weights, which are the part that has to be copied by hand.
    The ways of failing to build it are told apart because each has another
    cure: install the extra, reinstall it (torch is there and a DLL does not
    load, which Windows reports as ``OSError``, or its versions do not go
    together, a ``RuntimeError``), or take another file.

    Args:
        model_path: The checkpoint, or ``None`` for ``default_model_path()``.
        device: As ``TorchDetector``.
        score_threshold: As ``TorchDetector``.

    Raises:
        PluginRunError: The file is missing or cannot be looked for, torch is
            missing or does not load, or the checkpoint or the device is refused.
    """
    path = model_path if model_path is not None else default_model_path()
    try:
        is_file = path.is_file()
    except OSError as exc:
        # A folder the user cannot enter, a share that dropped: `is_file` raises
        # for what it cannot tell (FM-116), and the operator needs the same line.
        raise PluginRunError(
            _PLUGIN,
            f"cannot look for the model at {path} ({exc.strerror or exc}); set "
            f"SMC_MODELS to the folder that holds {DEFAULT_MODEL}, or give model_path",
        ) from exc
    if not is_file:
        raise PluginRunError(
            _PLUGIN,
            f"no model at {path}; put {DEFAULT_MODEL} there, set SMC_MODELS to "
            f"its folder, or give model_path",
        )
    try:
        return TorchDetector(path, device=device, score_threshold=score_threshold)
    except ImportError as exc:
        # Every ImportError, not ModuleNotFoundError alone: `DLL load failed while
        # importing _imaging` is one, and it is not a missing package.
        if isinstance(exc, ModuleNotFoundError):
            reason = f"torch is not installed ({exc}): install the detect extra: "
        else:
            reason = (
                f"torch could not be imported ({exc}): install or reinstall the "
                f"detect extra: "
            )
        raise PluginRunError(_PLUGIN, reason + "pip install -e .[detect]") from exc
    except (OSError, RuntimeError) as exc:
        # What is left of `OSError` is torch's own import (a DLL, on Windows): the
        # reading of the file is a `ValueError` by now. A torchvision of another
        # version than torch raises `RuntimeError`. torch is imported here, in the
        # process that already holds the stand's DLLs (FM-95), so a fresh console
        # tells a broken install from a conflict.
        raise PluginRunError(
            _PLUGIN,
            f"torch is installed but did not load ({exc}); reinstall the detect "
            f'extra. If `python -c "import torch"` works in a new console, the '
            f"stand's DLLs may conflict with torch: report it",
        ) from exc
    except ValueError as exc:
        raise PluginRunError(_PLUGIN, str(exc)) from exc
