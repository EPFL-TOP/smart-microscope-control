"""From a model's name to a detector, and what a result says of it: identity, and where it ran (inference.md §4, §5).

A model's identity is the SHA-256 of its weights and the digest of the
pre-processing the run applies, so that a change of either is another model
(ADR-0011): a resume refuses it and a result records it. ``load_model`` is the
one place a card becomes a ``Detector``: it looks for the weights, checks them
against the card before torch is imported, and says in plain words what to do
about each way it can fail.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import smc
from smc.inference.cards import ModelCard
from smc.inference.detector import Detector
from smc.inference.errors import ModelError
from smc.inference.local import load_local
from smc.inference.preprocessing import (
    ModelPreprocessing,
    preprocessing_for,
    preprocessing_sha256,
)

__all__ = [
    "ResolvedModel",
    "file_sha256",
    "load_model",
    "model_identity",
    "weights_folder",
]

#: The weights are hashed in pieces of this size: 166 MB are not read at once.
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A detector with what a result says of it: its name, its identity, where it runs."""

    name: str
    identity: str
    weights_sha256: str
    runs_on: str
    detector: Detector


def model_identity(weights_sha256: str, rule: ModelPreprocessing | None) -> str:
    """The weights' SHA-256 and the pre-processing's digest, joined by ``+``.

    The two halves stay readable, so that a resume can say which one changed.
    ``rule`` is the one the run applies (the card's, with any parameter in
    place), or ``None`` for a detector that has no pre-processing.
    """
    return f"{weights_sha256}+{preprocessing_sha256(rule)}"


def weights_folder() -> Path:
    """Where the weights are looked for: ``$SMC_MODELS`` or the checkout's ``local/models``.

    An empty or blank ``SMC_MODELS`` is no setting at all (FM-78): ``Path("")`` is
    the working directory, and the weights would be looked for wherever the
    command was typed. The checkout is the editable install's, which is how
    every PC runs smc.
    """
    folder = os.environ.get("SMC_MODELS", "").strip()
    if folder:
        return Path(folder)
    return Path(smc.__file__).resolve().parents[2] / "local" / "models"


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file, read in pieces: which weights these are.

    Raises:
        OSError: The file cannot be read; the caller says what to do about it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    card: ModelCard,
    *,
    preprocessing: ModelPreprocessing,
    frame_shape_px: tuple[int, int],
    frame_um_per_px: float,
    device: str,
    score_threshold: float,
    weights: Path | None = None,
) -> ResolvedModel:
    """The model a card names, running on this PC, or a ``ModelError`` that says what to do.

    The order is the cheap checks first: a pixel size the model is never given
    is refused before 166 MB of weights are looked for; the file is looked for
    and hashed before torch is imported, so a PC with neither the weights nor
    torch is told about the weights, which are the part that has to be copied by
    hand; and weights that are not the card's are refused without building the
    model. The classes of the checkpoint are compared with the card's last,
    because only the checkpoint can say them.

    Args:
        card: The model, by its card.
        preprocessing: The rule this run applies: the card's own, or with a
            parameter of ``cells`` in place of one part of it.
        frame_shape_px: The scan's frame shape, to work the input size out.
        frame_um_per_px: The scan's pixel size, for the same.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps`` (``TorchDetector``'s).
        score_threshold: Detections below it are dropped.
        weights: The weights' file, in place of ``card.weights`` in
            ``weights_folder()``; its SHA-256 must still be the card's.

    Raises:
        ModelError: The input size is refused, the file is missing, cannot be
            looked for or read, is not the card's weights, or torch or the
            checkpoint is refused; or the checkpoint's classes are not the
            card's.
    """
    try:
        shown = preprocessing_for(
            preprocessing,
            frame_shape_px=frame_shape_px,
            frame_um_per_px=frame_um_per_px,
        )
    except ValueError as exc:
        raise ModelError(
            f"model_um_per_px={preprocessing.model_um_per_px:g} cannot be used: {exc}"
            if preprocessing.model_um_per_px is not None
            else f"the frame cannot be shown to the model: {exc}"
        ) from exc

    path = weights
    try:
        if path is None:
            path = weights_folder() / card.weights
        is_file = path.is_file()
    except OSError as exc:
        # A folder the user cannot enter, a share that dropped: `is_file` raises
        # for what it cannot tell, and so does `resolve` on a checkout that sits on
        # one (FM-116); the operator needs the same line.
        where = path if path is not None else "the models folder"
        raise ModelError(
            f"cannot look for the model at {where} ({exc.strerror or exc}); set "
            f"SMC_MODELS to the folder that holds {card.weights}, or give model_path"
        ) from exc
    if not is_file:
        raise ModelError(
            f"no model at {path}; put {card.weights} there, set SMC_MODELS to "
            f"its folder, or give model_path"
        )

    try:
        sha = file_sha256(path)
    except OSError as exc:
        raise ModelError(
            f"{path.name} cannot be read to hash it ({exc.strerror or exc}); "
            f"is another program using it?"
        ) from exc
    if sha != card.weights_sha256:
        raise ModelError(
            f"{path.name} has the SHA-256 {sha}, and the card {card.name} names "
            f"{card.weights_sha256}: it is not the model the card describes; put "
            f"the card's file there, or add a card for this one"
        )

    detector = load_local(
        path, device=device, score_threshold=score_threshold, preprocessing=shown
    )
    if detector.classes != card.classes:
        raise ModelError(
            f"{path.name} has the classes {list(detector.classes)}, and the card "
            f"{card.name} names {list(card.classes)}"
        )
    return ResolvedModel(
        name=card.name,
        identity=model_identity(sha, preprocessing),
        weights_sha256=sha,
        runs_on=f"local:{detector.device}",
        detector=detector,
    )
