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

import contextlib
import logging
import math
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from smc.inference.cards import model_card, model_names
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, load_local_model
from smc.inference.served import ServedModel, encode_answer, encode_listing

__all__ = [
    "SERVER_SCORE_FLOOR",
    "ModelServer",
    "build_app",
    "check_serve_extra",
    "prepare",
    "serve",
]

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


def check_serve_extra() -> None:
    """Import what serving needs, so that a PC without the ``serve`` extra says so before a model is loaded.

    The command asks before it reads 166 MB of weights, not after. The kit's import
    takes seconds and loads ``cv2``, ``scipy`` and ``fastapi``: it is made here and in
    ``build_app``, never at module level, so that the acquisition PC and the fake model
    server import this module without any of it.

    Raises:
        ImportError: The kit or uvicorn is not installed, or does not import.
    """
    import imaging_server_kit  # noqa: F401
    import uvicorn  # noqa: F401


@contextlib.asynccontextmanager
async def _no_lifespan(app: object) -> AsyncIterator[None]:
    """A lifespan that does nothing: what the kit's own would do is talk to a hub we do not have."""
    yield


def build_app(server: ModelServer) -> object:
    """The kit's ASGI app for ``server``: two algorithms, ``models`` and ``detect``, as the wire says (§8.3).

    The kit's ``MultiAlgorithmServer`` registers with an "algorithm hub" at start-up
    (``requests.get("http://algorithm_hub:8000/")`` with no timeout) and deregisters,
    unguarded, at shutdown, so a server with no hub logs ``Application shutdown failed``
    (measured, 0.0.12). Its lifespan is replaced by one that does nothing.

    The decorator is called as a function, not written as ``@algorithm_server``: the kit
    is untyped, and a decorator that mypy cannot see makes the function untyped with the
    kit installed and an import error without it.

    Raises:
        ImportError: The kit is not installed (``check_serve_extra`` says so first).
    """
    from imaging_server_kit import (
        FloatSpinBoxUI,
        ImageUI,
        MultiAlgorithmServer,
        StringUI,
        algorithm_server,
    )

    def list_models() -> list[tuple[str, dict[str, str], str]]:
        return [(server.listing(), {"name": "models"}, "text")]

    def detect(
        model: str, image: npt.NDArray[Any], pixel_size_um: float
    ) -> list[tuple[str, dict[str, str], str]]:
        return [
            (server.detect(model, image, pixel_size_um), {"name": "detections"}, "text")
        ]

    models_algorithm = algorithm_server(
        parameters={},
        algorithm_name="models",
        title="Models",
        description="The models this server serves, with their identities.",
    )(list_models)
    detect_algorithm = algorithm_server(
        parameters={
            "model": StringUI(title="Model", description="A name of the listing."),
            "image": ImageUI(dimensionality=[2]),
            "pixel_size_um": FloatSpinBoxUI(
                title="Pixel size (um)",
                description="The frame's pixel size; 0 is unknown.",
                min=0.0,
                max=1000.0,
                default=0.0,
            ),
        },
        algorithm_name="detect",
        title="Detect",
        description="The detections of one 2-D frame by a model of the listing.",
    )(detect)
    multi = MultiAlgorithmServer("smc-models", [models_algorithm, detect_algorithm])
    multi.app.router.lifespan_context = _no_lifespan
    return multi.app


def serve(server: ModelServer, *, host: str, port: int) -> None:
    """Serve ``server`` over HTTP until it is interrupted: uvicorn's own graceful shutdown ends it.

    A first Ctrl-C ends uvicorn and this function returns, which is the server's normal
    end. Each call is logged by ``ModelServer`` at INFO (the model, the frame's shape
    and the seconds, never pixels); uvicorn prints a traceback on stderr for every 500
    (measured), which is the server's console and not a client's.

    Raises:
        ImportError: The kit or uvicorn is not installed.
    """
    import uvicorn

    uvicorn.run(build_app(server), host=host, port=port, log_level="info")


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
