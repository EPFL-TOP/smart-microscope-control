"""The look-up of the trained weights, until ``smc.inference`` names models by card (#684).

Everything that was here moved to ``smc.inference``; the names are re-exported
for the commit that moves their importers, and this module goes with the next
step.
"""

from __future__ import annotations

import os
from pathlib import Path

import smc
from smc.inference.blobs import BLOB_DETECTOR, BLOB_DETECTOR_IDENTITY, BlobDetector
from smc.inference.detector import Detection, Detector
from smc.inference.errors import ModelError
from smc.inference.local import (
    TorchDetector,
    detections_from_output,
    load_local,
)
from smc.inference.preprocessing import (
    DEFAULT_PREPROCESSING,
    MAX_INPUT_PX,
    MIN_INPUT_PX,
    Preprocessing,
    model_input_size_px,
    normalize_plane,
)
from smc.plugins.errors import PluginRunError

__all__ = [
    "BLOB_DETECTOR",
    "BLOB_DETECTOR_IDENTITY",
    "DEFAULT_MODEL",
    "DEFAULT_PREPROCESSING",
    "MAX_INPUT_PX",
    "MIN_INPUT_PX",
    "BlobDetector",
    "Detection",
    "Detector",
    "Preprocessing",
    "TorchDetector",
    "default_model_path",
    "detections_from_output",
    "load_detector",
    "model_input_size_px",
    "normalize_plane",
]

#: The trained single / doublet / debris model, trained by ``nikon-control``. It is
#: 166 MB and not in git: it lives in ``local/models`` (ignored) or in ``$SMC_MODELS``.
DEFAULT_MODEL = "cell_classes_model-cat-v2.pth"

_PLUGIN = "cells"


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


def load_detector(
    model_path: Path | None,
    *,
    device: str,
    score_threshold: float,
    preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
) -> Detector:
    """The trained detector, or a ``PluginRunError`` that says what to do about it.

    The file is looked for first, so a PC with neither the weights nor torch is
    told about the weights, which are the part that has to be copied by hand.

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
        return load_local(
            path,
            device=device,
            score_threshold=score_threshold,
            preprocessing=preprocessing,
        )
    except ModelError as exc:
        raise PluginRunError(_PLUGIN, str(exc)) from exc
