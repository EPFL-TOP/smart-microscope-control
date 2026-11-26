"""The model server: the models it loads, and what it answers for a frame (inference.md §8.6).

``ModelServer`` is the server **without HTTP**: ``listing()`` and ``detect()`` return the
JSON texts of ``smc.inference.served``'s wire. The fake model server of the tests is
this class behind ``ModelWire``, so that the two cannot drift. ``prepare`` loads the
cards a server serves.

The server runs this repository's own code: it reads the same cards and runs the same
``TorchDetector``, so the pre-processing is the same by construction, and it answers
with the model's identity, which the client compares with the one it worked out from
the card (ADR-0011). A server whose weights are not the card's, or that was started on
another checkout, is refused by the client, never believed.

This module imports nothing of the kit, torch, uvicorn or fastapi: they are imported
inside ``build_app`` and ``serve`` (a test walks the source for them), so that the
module imports on every PC and the fake of the tests can use ``ModelServer``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from smc.inference.cards import model_card, model_names
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, load_local_model
from smc.inference.served import ServedModel, encode_answer, encode_listing

__all__ = ["SERVER_SCORE_FLOOR", "ModelServer", "prepare"]

#: The server's detectors keep every detection (torchvision's own floor of 0.05
#: still applies); the **client** keeps ``score >= threshold``, the predicate of
#: ``detections_from_output``, so that a served run and a local one on the same
#: weights give the same detections, whatever threshold the run chose.
SERVER_SCORE_FLOOR = 0.0

#: A native rule never reads the frame's shape or pixel size (``preprocessing_for``
#: keeps the model's own resize), so the server, which serves native rules only and
#: has no frame when it loads a model, gives the loader these. They are placeholders,
#: named so; a sized rule is refused in ``prepare``.
_PLACEHOLDER_SHAPE_PX = (1, 1)
_PLACEHOLDER_UM_PER_PX = 1.0

#: A name the client sent is echoed in a refusal at this many characters at most.
_NAME_CHARS = 64

log = logging.getLogger("smc.inference.server")


class ModelServer:
    """The models a server serves, one forward pass at a time, answering as JSON texts.

    One ``threading.Lock`` makes the forward passes one at a time: the kit runs a
    function on a thread of its event loop's pool, and two calls would share the
    GPU. The errors are ``ValueError`` s that name no path (a failed call's
    message reaches the client, and from there a run folder, FM-50).

    ``models`` is read at every call and not copied: the fake model server hands
    it a live view whose names a test can hide.
    """

    def __init__(self, models: Mapping[str, ResolvedModel]) -> None:
        self._models = models
        self._lock = threading.Lock()

    def listing(self) -> str:
        """The models served, as the JSON text of a ``models`` result."""
        return encode_listing(
            [
                ServedModel(
                    name=name,
                    identity=model.identity,
                    classes=tuple(model.detector.classes),
                    device=model.detector.device,
                )
                for name, model in self._models.items()
            ]
        )

    def detect(self, model: str, frame: npt.NDArray[Any], pixel_size_um: float) -> str:
        """The detections of one frame by ``model``, as the JSON text of a ``detections`` result.

        Args:
            model: A name in the listing.
            frame: One 2-D frame, as the client sent it (dtype and values kept).
            pixel_size_um: The frame's pixel size; 0 is unknown. A native rule, the
                only kind served, does not read it.

        Raises:
            ValueError: The model is not served, the frame is not 2-D, or the pixel
                size is negative or not finite. The message names no path.
        """
        resolved = self._resolve(model)
        array = np.asarray(frame)
        if array.ndim != 2 or array.size == 0:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        if (
            isinstance(pixel_size_um, bool)
            or not math.isfinite(pixel_size_um)
            or pixel_size_um < 0.0
        ):
            raise ValueError("pixel_size_um must be a number of at least 0")
        with self._lock:
            started = time.perf_counter()
            detections = resolved.detector.detect(array)
            seconds = time.perf_counter() - started
        # Never the pixels: the frame is the lab's data.
        log.info("%s: a %dx%d frame in %.3f s", model, *array.shape, seconds)
        return encode_answer(
            model, resolved.identity, resolved.detector.device, detections
        )

    def _resolve(self, model: str) -> ResolvedModel:
        if not isinstance(model, str) or not model:
            raise ValueError("model is required")
        found = self._models.get(model)
        if found is None:
            served = ", ".join(self._models) or "no model"
            raise ValueError(
                f"the server does not serve {model[:_NAME_CHARS]!r}; it serves: {served}"
            )
        return found


def prepare(
    names: Sequence[str] | None,
    *,
    device: str,
    load: Callable[..., ResolvedModel] = load_local_model,
) -> tuple[ModelServer, list[str]]:
    """Load the cards a server serves, and what it skipped.

    ``load`` is ``load_local_model`` and never ``load_model``: a
    ``SMC_MODEL_SERVER`` in the server's own environment must not make it a client of
    itself. Each card is loaded with its own rule and the score floor.

    The server serves **native rules only**: a rule that names a pixel size needs a
    model per input size and an unbounded cache, which is a design change when a card
    needs it (inference.md §9). With no ``names`` every card is tried and one that
    cannot be served (no weights on this PC, another SHA-256, a sized rule) is
    skipped, its reason in the notes; with ``names`` any failure is a ``ModelError``.
    The notes may name paths: they go to the server's own console, never to a client.

    Args:
        names: The cards to serve, or ``None`` (or none at all) for every card.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``, as ``TorchDetector`` takes it.
        load: The loader; a test passes a recorder.

    Returns:
        The server, and one note per card that was skipped.

    Raises:
        ModelError: A named card cannot be loaded or served (its diagnosis), or no
            model at all can be served.
    """
    wanted = tuple(dict.fromkeys(names)) if names else None
    notes: list[str] = []
    models: dict[str, ResolvedModel] = {}
    for name in model_names() if wanted is None else wanted:
        try:
            models[name] = _load_native(name, device=device, load=load)
        except ModelError as exc:
            if wanted is not None:
                raise
            notes.append(f"{name} is not served: {exc}")
    if not models:
        reason = "; ".join(notes) if notes else "there is no model card"
        raise ModelError(f"no model can be served here: {reason}")
    return ModelServer(models), notes


def _load_native(
    name: str, *, device: str, load: Callable[..., ResolvedModel]
) -> ResolvedModel:
    """One card, loaded with its own native rule; a rule that names a pixel size is refused."""
    card = model_card(name)
    size = card.preprocessing.model_um_per_px
    if size is not None:
        raise ModelError(
            f"the card {name} shows frames to the model at {size:g} um per pixel, and "
            f"the server serves the model's own resize only; run it on the acquisition "
            f"PC (unset SMC_MODEL_SERVER) or add the sized rule to the server's design"
        )
    return load(
        card,
        preprocessing=card.preprocessing,
        frame_shape_px=_PLACEHOLDER_SHAPE_PX,
        frame_um_per_px=_PLACEHOLDER_UM_PER_PX,
        device=device,
        score_threshold=SERVER_SCORE_FLOOR,
        weights=None,
    )
