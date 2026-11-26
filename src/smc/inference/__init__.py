"""Models by name, behind one detector interface (docs/design/inference.md, ADR-0011).

``Detector`` and ``Detection`` are what a stage asks of a model; a **card** in
``cards/`` names a model by its weights, classes and pre-processing; and
``load_model`` turns a card into a ``ResolvedModel`` that says what a result
records: the name, the identity and where it ran.

Importing this package loads neither torch nor scikit-image: ``TorchDetector``
(``smc.inference.local``) imports torch inside its constructor, and
``BlobDetector`` and ``blob_model`` (``smc.inference.blobs``) are imported from
their module. It imports nothing of ``smc.plugins``, ``smc.hardware``,
``smc.testing`` or ``smc.ui``: it is part of the algorithm developer's kit and
``cells`` depends on it, not the reverse.
"""

from __future__ import annotations

from smc.inference.cards import (
    DEFAULT_MODEL_NAME,
    ModelCard,
    model_card,
    model_cards,
    model_names,
)
from smc.inference.detector import Detection, Detector
from smc.inference.errors import ModelError
from smc.inference.models import (
    ResolvedModel,
    load_model,
    model_identity,
    weights_folder,
)
from smc.inference.preprocessing import ModelPreprocessing, Preprocessing

__all__ = [
    "DEFAULT_MODEL_NAME",
    "Detection",
    "Detector",
    "ModelCard",
    "ModelError",
    "ModelPreprocessing",
    "Preprocessing",
    "ResolvedModel",
    "load_model",
    "model_card",
    "model_cards",
    "model_identity",
    "model_names",
    "weights_folder",
]
