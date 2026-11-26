"""The trained model run on this PC: ``TorchDetector`` and how its output becomes detections (m3 §9.1).

torchvision emits ``[x0, y0, x1, y1]``; the swap to rows first is made once, in
``detections_from_output``, so nothing downstream thinks about it.

torch and torchvision are the ``detect`` extra and are imported **inside**
``TorchDetector.__init__`` only: this module imports on a PC that never
installed them, so ``smc run --list`` shows ``cells`` everywhere and a missing
torch is an instruction, not a crash.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.preprocessing import (
    DEFAULT_PREPROCESSING,
    Preprocessing,
    normalize_plane,
)

__all__ = ["TorchDetector", "detections_from_output", "load_local"]

_BACKGROUND = "__background__"
_PREDICTOR_KEY = "roi_heads.box_predictor.cls_score.weight"

log = logging.getLogger("smc.inference.local")


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
        preprocessing: The intensity percentiles and the size the frame is resized to
            before the network. The default is how the model was trained: its own
            resize (800 and 1333) and the 0.5 and 99.5 percentiles. The resize is
            built into the model, so it is set here and not per frame.

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

    def __init__(
        self,
        weights: Path,
        *,
        device: str,
        score_threshold: float,
        preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
    ) -> None:
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
        # The resize has no weights, so the checkpoint loads strictly whatever the
        # size. Only an input size that was asked for is passed: without one the
        # model keeps the 800 and 1333 it was trained with.
        size: dict[str, int] = {}
        if preprocessing.input_size_px is not None:
            size = {
                "min_size": preprocessing.input_size_px[0],
                "max_size": preprocessing.input_size_px[1],
            }
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(names), **size
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
        self.preprocessing = preprocessing
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

        One frame per forward pass, under ``no_grad``, resized by the model as
        ``preprocessing`` set it: batching frames is not decided here. The plane
        is scaled with ``preprocessing.percentiles`` and repeated to three
        channels, which is what the backbone expects.

        Raises:
            ValueError: ``frame`` is not 2-D (a colour frame is not a brightfield one).
        """
        array = np.asarray(frame)
        if array.ndim != 2:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        torch = self._torch
        plane = torch.from_numpy(normalize_plane(array, self.preprocessing.percentiles))
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


def load_local(
    weights: Path,
    *,
    device: str,
    score_threshold: float,
    preprocessing: Preprocessing,
) -> TorchDetector:
    """The trained detector built from ``weights``, or a ``ModelError`` that says what to do.

    The ways of failing to build it are told apart because each has another
    cure: install the extra, reinstall it (torch is there and a DLL does not
    load, which Windows reports as ``OSError``, or its versions do not go
    together, a ``RuntimeError``), or take another file. The file is looked for
    and checked against its card before this is called (``load_model``), so a
    PC with neither the weights nor torch is told about the weights, which are
    the part that has to be copied by hand.

    Args:
        weights: The checkpoint.
        device: As ``TorchDetector``.
        score_threshold: As ``TorchDetector``.
        preprocessing: As ``TorchDetector``.

    Raises:
        ModelError: torch is missing or does not load, or the checkpoint or the
            device is refused.
    """
    try:
        return TorchDetector(
            weights,
            device=device,
            score_threshold=score_threshold,
            preprocessing=preprocessing,
        )
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
        raise ModelError(reason + "pip install -e .[detect]") from exc
    except (OSError, RuntimeError) as exc:
        # What is left of `OSError` is torch's own import (a DLL, on Windows): the
        # reading of the file is a `ValueError` by now. A torchvision of another
        # version than torch raises `RuntimeError`. torch is imported here, in the
        # process that already holds the stand's DLLs (FM-95), so a fresh console
        # tells a broken install from a conflict.
        raise ModelError(
            f"torch is installed but did not load ({exc}); reinstall the detect "
            f'extra. If `python -c "import torch"` works in a new console, the '
            f"stand's DLLs may conflict with torch: report it"
        ) from exc
    except ValueError as exc:
        raise ModelError(str(exc)) from exc
