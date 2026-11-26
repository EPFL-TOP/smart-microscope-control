"""A model server with no socket: the real ``ServedDetector`` and ``cells`` run against it (inference.md §8.8).

``FakeModelServer`` is to the model server what ``FakeMTB`` is to the Zeiss components:
it speaks the wire as the kit's server does (404 for another path, 422 for a body with
another key than the three or a pixel size below 0, 500 for what the model or the image
raises, 201 with the result list), so the code under test is the production client, and
it **is** ``ModelServer`` behind ``ModelWire``, so the answers cannot drift from the
real server's. Its failures are knobs a test sets (a server that is absent, slow,
failing, or another model), and it **records what it was asked**, so that a test asserts
on the body the client built and not on the fake's own opinion of it (FM-260).

Nothing sleeps: a latency above the call's timeout is a timeout, at once, and no test
here depends on how fast anything runs (FM-44).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy.typing as npt

from smc.inference import served
from smc.inference.models import ResolvedModel
from smc.inference.served import (
    ModelWire,
    ServedAnswer,
    decode_answer,
    decode_frame,
    decode_listing,
    encode_answer,
    encode_listing,
    error_from_status,
    timed_out,
    unreachable,
)
from smc.inference.server import ModelServer

if TYPE_CHECKING:
    import pytest

__all__ = ["FakeModelServer", "ServedCall"]

_DETECT = "/detect/process"
_MODELS = "/models/process"


@dataclass(frozen=True, slots=True, eq=False)
class ServedCall:
    """One call the fake received: the path, the body exactly as built, the frame in it, the timeout."""

    path: str
    body: dict[str, object]
    #: The array the body's image decodes to, or ``None`` when there is none.
    frame: npt.NDArray[Any] | None
    read_timeout_s: float


class _Visible(Mapping[str, ResolvedModel]):
    """The models the fake serves: those it holds, but the names a test hid."""

    def __init__(
        self, models: Mapping[str, ResolvedModel], hidden: Callable[[], set[str]]
    ) -> None:
        self._models = models
        self._hidden = hidden

    def __getitem__(self, name: str) -> ResolvedModel:
        if name in self._hidden():
            raise KeyError(name)
        return self._models[name]

    def __iter__(self) -> Iterator[str]:
        return (name for name in self._models if name not in self._hidden())

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _NoServerThere:
    """The wire to an address the fake is not at: nobody answers (a refusal, never a network call)."""

    def post(
        self, path: str, body: Mapping[str, object], *, read_timeout_s: float
    ) -> object:
        raise unreachable("connection refused")


class FakeModelServer:
    """A ``ModelWire`` over a ``ModelServer``, answering as the kit's server does, with failures a test sets.

    Knobs, plain attributes a test sets at any time (``on_call`` is where a test
    sets one in the middle of a run):

    * ``absent``: every call is ``ServedError("unreachable")`` and the server never
      saw it (it is not in ``calls``).
    * ``latency_s``: a call whose ``read_timeout_s`` is below it is
      ``ServedError("timeout")``; nothing sleeps. The call arrived, so it is in
      ``calls``, and the model did not run.
    * ``fail_next``: the next n calls are ``ServedError("error")`` (``HTTP 500``).
    * ``identity_override``: the listing and the answers carry this identity, as
      a server started on another checkout or other weights would.
    * ``hidden``: names the server does not serve (not listed, refused as unknown).
    * ``on_call``: called with each call as it arrives, **before** it is answered:
      a test that sets ``absent`` there has that very call refused.

    ``calls`` records every call that reached the server, in order: the path, the
    exact body the client built, the frame decoded from it and the read timeout.
    """

    def __init__(
        self,
        models: Mapping[str, ResolvedModel],
        *,
        address: str = "http://models.test:8000",
    ) -> None:
        self.address = address
        self.absent = False
        self.latency_s = 0.0
        self.fail_next = 0
        self.identity_override: str | None = None
        self.hidden: set[str] = set()
        self.on_call: Callable[[ServedCall], None] | None = None
        self.calls: list[ServedCall] = []
        self._server = ModelServer(_Visible(models, lambda: self.hidden))

    def __repr__(self) -> str:
        return "FakeModelServer()"

    def post(
        self, path: str, body: Mapping[str, object], *, read_timeout_s: float
    ) -> object:
        """The result list of a 201, or the ``ServedError`` the real client would raise for the failure set."""
        if self.absent:
            raise unreachable("connection refused")
        image = body.get("image")
        call = ServedCall(
            path=path,
            body=dict(body),
            frame=_decoded(image) if isinstance(image, str) else None,
            read_timeout_s=read_timeout_s,
        )
        self.calls.append(call)
        if self.on_call is not None:
            self.on_call(call)
        if self.absent:
            raise unreachable("connection refused")
        if self.latency_s > read_timeout_s:
            raise timed_out(read_timeout_s)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise error_from_status(
                500,
                {"error_type": "internal_server_error", "message": "injected failure"},
            )
        if path == _MODELS:
            text, name = self._listing(body), "models"
        elif path == _DETECT:
            text, name = self._detect(body), "detections"
        else:
            raise error_from_status(404, {"detail": _not_found(path)})
        return [{"type": "text", "data": text, "data_params": {"name": name}}]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make ``load_model`` go to this server: the variable is set and the wire is this fake.

        ``served.open_wire`` is replaced by a function that returns the fake for
        its own address and a wire that nobody answers for any other, so a test
        that set another address sees a refusal and never a network call. A test
        that forgot to call this sees ``load_model`` run the model here.
        """
        monkeypatch.setenv(served.SERVER_ENV, self.address)

        def open_fake(address: str) -> ModelWire:
            return self if address == self.address else _NoServerThere()

        monkeypatch.setattr(served, "open_wire", open_fake)

    # --- what the kit's server does with a call ---------------------------------

    def _listing(self, body: Mapping[str, object]) -> str:
        if body:
            raise error_from_status(422, {"detail": [_extra(sorted(body)[0])]})
        text = self._server.listing()
        if self.identity_override is None:
            return text
        models = decode_listing(text)
        return encode_listing(
            [replace(m, identity=self.identity_override) for m in models]
        )

    def _detect(self, body: Mapping[str, object]) -> str:
        model, image, pixel_size_um = _validated(body)
        try:
            frame = decode_frame(image)
        except ValueError as exc:
            # The kit's own 422 handler fails to serialise this one: a 500 (measured).
            raise error_from_status(
                500, {"error_type": "internal_server_error", "message": str(exc)}
            ) from None
        try:
            text = self._server.detect(model, frame, pixel_size_um)
        except Exception as exc:
            # A function that raises is a 500 with its message, whatever it raised.
            raise error_from_status(
                500, {"error_type": "internal_server_error", "message": str(exc)}
            ) from None
        if self.identity_override is None:
            return text
        answer: ServedAnswer = decode_answer(text)
        return encode_answer(
            answer.model, self.identity_override, answer.device, answer.detections
        )


def _decoded(image: str) -> npt.NDArray[Any] | None:
    try:
        return decode_frame(image)
    except ValueError:
        return None


def _not_found(path: str) -> str:
    parts = path.strip("/").split("/")
    if len(parts) == 2 and parts[1] == "process":
        return f"Algorithm {parts[0]} not found"
    return "Not Found"


def _extra(key: str) -> dict[str, object]:
    return {
        "type": "extra_forbidden",
        "loc": ["body", key],
        "msg": "Extra inputs are not permitted",
    }


def _validated(body: Mapping[str, object]) -> tuple[str, str, float]:
    """The three parameters of ``detect`` as the kit validates them: another key, a missing one, a wrong type, a pixel size below 0 are a 422."""
    problems: list[dict[str, object]] = []
    for key in sorted(set(body) - {"model", "image", "pixel_size_um"}):
        problems.append(_extra(key))
    for key in ("model", "image", "pixel_size_um"):
        if key not in body:
            problems.append(
                {"type": "missing", "loc": ["body", key], "msg": "Field required"}
            )
    model, image, size = body.get("model"), body.get("image"), body.get("pixel_size_um")
    if "model" in body and not isinstance(model, str):
        problems.append(_wrong("model", "Input should be a valid string"))
    if "image" in body and not isinstance(image, str):
        problems.append(_wrong("image", "Input should be a valid string"))
    if "pixel_size_um" in body:
        if isinstance(size, bool) or not isinstance(size, (int, float)):
            problems.append(_wrong("pixel_size_um", "Input should be a valid number"))
        elif not size >= 0:
            problems.append(
                _wrong("pixel_size_um", "Input should be greater than or equal to 0")
            )
    if problems:
        raise error_from_status(422, {"detail": problems})
    assert isinstance(model, str)
    assert isinstance(image, str)
    assert isinstance(size, (int, float))
    return model, image, float(size)


def _wrong(key: str, message: str) -> dict[str, object]:
    return {"type": "value_error", "loc": ["body", key], "msg": message}
