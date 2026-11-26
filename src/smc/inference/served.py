"""A model that runs on a server: the kit's wire, the client that speaks it, and what a failed call says (inference.md §8).

The server is this repository's own ``smc serve-models`` (``smc.inference.server``),
built on ``imaging-server-kit``; the client here speaks the kit's ``/process`` wire
with the standard library alone. The kit has no timeout in any 0.0.x client and its
import takes six seconds and loads ``cv2``, ``scipy`` and ``fastapi``, which the
acquisition PC, with the stand's DLLs, has no business loading (FM-95); so the kit is
imported **only** in ``server.py``, inside functions, and the wire's constants and
codecs live here, shared by the client, the server and the fake model server.

Three rules shape every line of the client. **Every call has a timeout** (FM-31):
connect, check and detect each have their own, and a host that drops packets holds
the stage for seconds, not for the operating system's minutes. **No proxy, ever**
(FM-152): the server is reached on the lab's network only (ADR-0011), and
``http.client`` never reads the proxy variables. **No message holds the address, the
host, the port or the URL** (FM-50): a failed stage's diagnosis is kept in the run
folder, and run documents are attached to issues. A message is built from the kind
of failure, the status and the server's own text, never from ``str(exc)`` of a
library, which puts the URL in it.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import logging
import math
import os
import socket
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt
import tifffile

from smc.inference.cards import ModelCard
from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, model_identity
from smc.inference.preprocessing import ModelPreprocessing

__all__ = [
    "CHECK_TIMEOUT_S",
    "CONNECT_TIMEOUT_S",
    "DETECT_TIMEOUT_S",
    "MAX_ANSWER_BYTES",
    "SCHEMA",
    "SERVER_ENV",
    "TIMEOUT_ENV",
    "HttpWire",
    "ModelWire",
    "ServedAnswer",
    "ServedCheck",
    "ServedDetector",
    "ServedError",
    "ServedKind",
    "ServedModel",
    "check_served",
    "decode_answer",
    "decode_frame",
    "decode_listing",
    "encode_answer",
    "encode_frame",
    "encode_listing",
    "error_from_status",
    "load_served",
    "open_wire",
    "result_text",
    "timed_out",
    "unreachable",
]

#: http://host:port; blank is unset (FM-78).
SERVER_ENV = "SMC_MODEL_SERVER"
#: The read timeout of a detect call, in seconds; blank is unset.
TIMEOUT_ENV = "SMC_MODEL_TIMEOUT_S"
#: Assumed: a LAN host that is up answers at once. It stays above the two seconds a
#: refused loopback connection takes on Windows, so that a closed port is "refused",
#: not "no answer in time".
CONNECT_TIMEOUT_S = 3.0
#: Assumed: the listing is a few hundred bytes.
CHECK_TIMEOUT_S = 10.0
#: Assumed: 0.5 to 0.8 s a frame on a CPU, and a cold first call loads kernels. The
#: lab's first measurement of a call on its network sets it for good (#686).
DETECT_TIMEOUT_S = 60.0
#: A listing or an answer is a few kB; more than this is not the server's.
MAX_ANSWER_BYTES = 4 * 1024 * 1024
#: The longest ``SMC_MODEL_TIMEOUT_S`` accepted: a day. A larger one overflows the socket's
#: own timeout on some platforms, and no call is waited for that long.
_MAX_TIMEOUT_S = 24 * 3600.0
#: The version of the two documents below; a server of another one is "malformed".
SCHEMA = 1

#: The server's own text is kept to a line of this many characters.
_SERVER_TEXT_CHARS = 200
#: A name or an identity the server sent is repeated in a message at this many characters.
_ECHO_CHARS = 64
#: The characters of each half of an identity a message shows: enough to tell two apart.
_HALF_CHARS = 12
_BAD_ADDRESS = f"{SERVER_ENV} is not an address like http://host:8000"

log = logging.getLogger("smc.inference.served")

ServedKind = Literal["unreachable", "timeout", "error", "identity", "malformed"]


class ServedError(ModelError):
    """A call to the model server that did not give a usable answer; ``kind`` says which.

    ``unreachable``: no connection (refused, unroutable, no name, none within
    ``CONNECT_TIMEOUT_S``), or the connection broke mid-call. ``timeout``:
    connected, and no answer within the call's read timeout. ``error``: the server
    answered a status other than 201. ``identity``: the answer is for another model
    than the one asked for, or carries another identity. ``malformed``: not JSON,
    another schema, a field of the wrong type, a number that is not finite, more
    than ``MAX_ANSWER_BYTES``. The message names none of the address; it is only
    the text and the kind (a failure that is kept holds no exception, FM-98).
    """

    kind: ServedKind

    def __init__(self, kind: ServedKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind

    def __reduce__(self) -> tuple[Any, ...]:
        # The default would rebuild it from `args` alone, which is the message.
        return (type(self), (self.kind, str(self)))


@dataclass(frozen=True, slots=True)
class ServedModel:
    """One model the server lists: its name, identity, classes and device."""

    name: str
    identity: str
    classes: tuple[str, ...]
    device: str


@dataclass(frozen=True, slots=True)
class ServedAnswer:
    """What the server says of one frame: the model it ran, its identity and device, the boxes."""

    model: str
    identity: str
    device: str
    detections: tuple[Detection, ...]


class ModelWire(Protocol):
    """A way to post one JSON document to the model server: HTTP, or the fake of the tests."""

    def post(
        self, path: str, body: Mapping[str, object], *, read_timeout_s: float
    ) -> object:
        """The parsed JSON of a 201 answer; a ``ServedError`` for anything else."""
        ...


# --- the errors, worded once so that the client and the fake cannot drift ---------


#: What to check when nobody answers: the server is not running, or the address is wrong.
_ADVICE_NO_ANSWER = (
    f"is smc serve-models running there, and is {SERVER_ENV} its address?"
)
#: A TLS failure with a server that is running: the cure is the scheme or the certificate.
_ADVICE_TLS = (
    "use http:// unless the server serves TLS, and https:// only with a "
    "certificate this PC trusts"
)
#: A connection that was made and then broke: the server is there, and probably restarted.
_ADVICE_BROKE = "the server was reached and ended the call: look at its console"


def unreachable(reason: str, *, advice: str = _ADVICE_NO_ANSWER) -> ServedError:
    """The server cannot be reached: ``reason`` is the stack's own words, never the address.

    ``advice`` is the cure for that reason: the same sentence for a TLS handshake that
    failed against a server that is running, and for a connection that was refused,
    would send the operator to the wrong place (FM-263).
    """
    return ServedError(
        "unreachable", f"the model server does not answer ({reason}); {advice}"
    )


def timed_out(read_timeout_s: float) -> ServedError:
    """The server was reached and did not answer within the call's read timeout."""
    return ServedError(
        "timeout", f"the model server did not answer in {read_timeout_s:g} s"
    )


def _first_line(text: str) -> str:
    """The first non-empty line of the server's own text, printable, at most 200 characters."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            cleaned = "".join(c if c.isprintable() else " " for c in stripped)
            return cleaned[:_SERVER_TEXT_CHARS]
    return ""


def _server_text(payload: object) -> str:
    """The server's words in an error body: the kit's ``message`` or ``detail`` (a string, or a list of problems)."""
    if not isinstance(payload, dict):
        return ""
    message = payload.get("message")
    if isinstance(message, str):
        return _first_line(message)
    detail = payload.get("detail")
    if isinstance(detail, str):
        return _first_line(detail)
    if isinstance(detail, list):
        # A validation error lists its problems with the input that failed, which is
        # a frame: only the first one's location and message are taken.
        for item in detail:
            if isinstance(item, dict) and isinstance(item.get("msg"), str):
                loc = item.get("loc")
                where = (
                    ".".join(str(part) for part in loc if isinstance(part, (str, int)))
                    if isinstance(loc, list)
                    else ""
                )
                text = f"{where}: {item['msg']}" if where else str(item["msg"])
                return _first_line(text)
    return ""


def error_from_status(status: int, payload: object) -> ServedError:
    """The ``error`` for a status other than 201, with the server's first line of text.

    ``payload`` is the parsed body or ``None``; the fake model server passes the
    body the kit would send, so that the client's words are tested on both.
    """
    said = _server_text(payload)
    suffix = f"; the server said: {said}" if said else ""
    return ServedError("error", f"HTTP {status} from the model server{suffix}")


def _malformed(what: str, detail: str) -> ServedError:
    return ServedError("malformed", f"the server's {what} is malformed: {detail}")


# --- the codecs -------------------------------------------------------------------


def encode_frame(frame: npt.NDArray[Any]) -> str:
    """A frame as the kit sends an image: a TIFF of the array as it is, base64, ASCII.

    The dtype and the values are kept (the server's detectors see what a local one
    would); a frame that is a view is made contiguous first.
    """
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, np.ascontiguousarray(frame))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_frame(text: str) -> npt.NDArray[Any]:
    """The array of an ``encode_frame`` text.

    Raises:
        ValueError: ``text`` is not base64 of a TIFF; the message says which.
    """
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError as exc:  # binascii.Error is one
        raise ValueError("the image is not base64") from exc
    try:
        return tifffile.imread(io.BytesIO(raw))
    except (tifffile.TiffFileError, OSError, KeyError, IndexError) as exc:
        # tifffile's own reader error derives from Exception on the older versions
        # that Python 3.10 installs, and a truncated file can be any of the others.
        raise ValueError("the image is not a TIFF") from exc


def encode_listing(models: Sequence[ServedModel]) -> str:
    """The server's listing as the JSON text of a ``models`` result (inference.md §8.3)."""
    return json.dumps(
        {
            "schema": SCHEMA,
            "models": [
                {
                    "name": m.name,
                    "identity": m.identity,
                    "classes": list(m.classes),
                    "device": m.device,
                }
                for m in models
            ],
        },
        allow_nan=False,
    )


def encode_answer(
    model: str, identity: str, device: str, detections: Sequence[Detection]
) -> str:
    """One frame's answer as the JSON text of a ``detections`` result.

    A float is Python's ``repr`` in the JSON, so a detection crosses the wire
    exactly; a box is ``[row0, col0, row1, col1, label, score]``.
    """
    return json.dumps(
        {
            "schema": SCHEMA,
            "model": model,
            "identity": identity,
            "device": device,
            "detections": [
                [d.row0_px, d.col0_px, d.row1_px, d.col1_px, d.label, d.score]
                for d in detections
            ],
        },
        allow_nan=False,
    )


def _no_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not a number")


def _loads(raw: str | bytes) -> object:
    """The JSON of ``raw``; ``NaN`` and ``Infinity``, which Python reads, are not JSON.

    Raises:
        ValueError: Not JSON (a ``UnicodeDecodeError`` is one), or too deeply nested.
    """
    try:
        return json.loads(raw, parse_constant=_no_constant)
    except RecursionError as exc:
        raise ValueError("nested too deeply") from exc


def _document(text: str, what: str) -> dict[str, object]:
    """The JSON object of a listing or an answer, of the schema this client reads."""
    try:
        data = _loads(text)
    except ValueError:
        raise _malformed(what, "it is not JSON") from None
    if not isinstance(data, dict):
        raise _malformed(what, "it is not a JSON object")
    schema = data.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != SCHEMA:
        raise _malformed(what, f"it is not of schema {SCHEMA}")
    return data


def _string(data: Mapping[str, object], key: str, what: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise _malformed(what, f"{key} is not a string")
    return value


def _number(value: object, what: str, key: str) -> float:
    """A finite number: a ``bool`` is not one, and ``1e999`` reads as infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _malformed(what, f"{key} is not a number")
    try:
        number = float(value)
    except OverflowError:
        # An integer of more than 308 digits: JSON allows it, a float cannot hold it.
        raise _malformed(what, f"{key} is not finite") from None
    if not math.isfinite(number):
        raise _malformed(what, f"{key} is not finite")
    return number


def decode_listing(text: str) -> tuple[ServedModel, ...]:
    """The models a listing names.

    Raises:
        ServedError: ``malformed``: not JSON, another schema, a field of the wrong type.
    """
    what = "listing"
    data = _document(text, what)
    raw = data.get("models")
    if not isinstance(raw, list):
        raise _malformed(what, "models is not a list")
    models: list[ServedModel] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _malformed(what, "a model is not an object")
        classes = item.get("classes")
        if not isinstance(classes, list) or not all(
            isinstance(label, str) for label in classes
        ):
            raise _malformed(what, "classes is not a list of strings")
        models.append(
            ServedModel(
                name=_string(item, "name", what),
                identity=_string(item, "identity", what),
                classes=tuple(classes),
                device=_string(item, "device", what),
            )
        )
    return tuple(models)


def decode_answer(text: str) -> ServedAnswer:
    """One frame's answer.

    Raises:
        ServedError: ``malformed``: not JSON, another schema, a field of the wrong
            type, a number that is not finite, a detection that is not six values.
    """
    what = "answer"
    data = _document(text, what)
    raw = data.get("detections")
    if not isinstance(raw, list):
        raise _malformed(what, "detections is not a list")
    detections: list[Detection] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 6:
            raise _malformed(what, "a detection is not six values")
        label = item[4]
        if not isinstance(label, str):
            raise _malformed(what, "a label is not a string")
        row0, col0, row1, col1, _label, score = item
        detections.append(
            Detection(
                _number(row0, what, "a box coordinate"),
                _number(col0, what, "a box coordinate"),
                _number(row1, what, "a box coordinate"),
                _number(col1, what, "a box coordinate"),
                label,
                _number(score, what, "a score"),
            )
        )
    return ServedAnswer(
        model=_string(data, "model", what),
        identity=_string(data, "identity", what),
        device=_string(data, "device", what),
        detections=tuple(detections),
    )


def result_text(result: object, what: str) -> str:
    """The first ``text`` result of a kit answer: the JSON document the server wrote.

    A kit answer is a list of ``{"type", "data", "data_params"}``; the models and
    detect algorithms return one result of type ``text``.

    Raises:
        ServedError: ``malformed``: no result of type ``text`` that holds a string.
    """
    if isinstance(result, list):
        for item in result:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("data"), str)
            ):
                return str(item["data"])
    raise _malformed(what, "it holds no text result")


# --- the wire ---------------------------------------------------------------------


def _parse_address(address: str) -> tuple[str, int, bool]:
    """Host, port and whether it is TLS, of ``http://host:port`` or ``https://host:port``.

    No user, no path, an explicit port: the one shape the lab's operator writes
    (ADR-0011). Every refusal is one fixed message, **without the value**: a
    mistyped address with a password in it must not reach a run folder (FM-50).

    A space or a control character anywhere is refused too. ``urlsplit`` accepts it
    in a host (``http://10.0.0.5 :8000``) and ``http.client`` then raises
    ``InvalidURL``, with the host in its text, when the connection is built: outside
    every ``try`` of the wire, so it would be neither a ``ModelError`` nor a
    ``ServedError`` and would reach the run folder's diagnosis (FM-277).

    Raises:
        ModelError: ``address`` is not of that shape.
    """
    if any(char <= " " or char == "\x7f" for char in address):
        raise ModelError(_BAD_ADDRESS)
    try:
        parts = urlsplit(address)
        host = parts.hostname
        port = parts.port
    except ValueError:
        # An invalid IPv6 literal, a port out of range: the exception's text holds
        # part of the address, so it is dropped with its context.
        raise ModelError(_BAD_ADDRESS) from None
    if (
        parts.scheme not in ("http", "https")
        or not host
        or port is None
        or port < 1
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ModelError(_BAD_ADDRESS)
    return host, port, parts.scheme == "https"


def _os_reason(exc: OSError | ValueError) -> str:
    """Why a connection failed, in words that do not hold the address.

    A TLS error's own text can name the host the certificate was checked against,
    so those are fixed words; the others are the operating system's, which name
    the condition and not the peer.
    """
    if isinstance(exc, ssl.SSLError):
        return "the TLS handshake failed"
    if isinstance(exc, ConnectionRefusedError):
        return "connection refused"
    if isinstance(exc, socket.gaierror):
        return "the host name cannot be resolved"
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return "the address cannot be used"


class HttpWire:
    """The kit's ``/process`` wire over ``http.client``: one connection per call, a timeout on each step.

    ``http.client`` and nothing else: it never reads a proxy variable (FM-152) and
    has no import cost. The connection is made with ``CONNECT_TIMEOUT_S`` and the
    socket's timeout is then set to the call's own read timeout, which is per
    socket operation and bounds the call because the server writes its answer, a
    few kB, in one piece once it has computed it. The address is never kept in a
    message, in the ``repr`` or in an exception chain.
    """

    def __init__(self, address: str) -> None:
        self._host, self._port, self._tls = _parse_address(address)

    def __repr__(self) -> str:
        return "HttpWire()"

    def post(
        self, path: str, body: Mapping[str, object], *, read_timeout_s: float
    ) -> object:
        """The parsed JSON of a 201 answer to ``POST path``.

        Raises:
            ServedError: ``unreachable``, ``timeout``, ``error`` (a status other than
                201, with the server's first line) or ``malformed`` (not JSON, or
                more than ``MAX_ANSWER_BYTES``).
        """
        payload = json.dumps(body, allow_nan=False).encode("ascii")
        connection_type = (
            http.client.HTTPSConnection if self._tls else http.client.HTTPConnection
        )
        connection = connection_type(self._host, self._port, timeout=CONNECT_TIMEOUT_S)
        try:
            return self._exchange(connection, path, payload, read_timeout_s)
        finally:
            connection.close()

    def _exchange(
        self,
        connection: http.client.HTTPConnection,
        path: str,
        payload: bytes,
        read_timeout_s: float,
    ) -> object:
        try:
            connection.connect()
        except TimeoutError:
            raise unreachable(f"no connection within {CONNECT_TIMEOUT_S:g} s") from None
        except (OSError, ValueError) as exc:
            # A ValueError is a host name that cannot be encoded (a label over 63
            # characters is a `UnicodeError`).
            raise unreachable(
                _os_reason(exc),
                advice=_ADVICE_TLS
                if isinstance(exc, ssl.SSLError)
                else _ADVICE_NO_ANSWER,
            ) from None
        if connection.sock is not None:
            connection.sock.settimeout(read_timeout_s)
        try:
            connection.request(
                "POST",
                path,
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(MAX_ANSWER_BYTES + 1)
        except TimeoutError:
            raise timed_out(read_timeout_s) from None
        except http.client.RemoteDisconnected:
            raise unreachable(
                "the server closed the connection", advice=_ADVICE_BROKE
            ) from None
        except (OSError, http.client.HTTPException):
            raise unreachable("the connection broke", advice=_ADVICE_BROKE) from None
        if response.status != 201:
            # Whatever the body is: a body over the limit is not parsed for its text.
            parsed: object = None
            if len(raw) <= MAX_ANSWER_BYTES:
                try:
                    parsed = _loads(raw)
                except ValueError:
                    parsed = None
            raise error_from_status(response.status, parsed)
        if len(raw) > MAX_ANSWER_BYTES:
            raise ServedError(
                "malformed",
                f"the server's answer is longer than {MAX_ANSWER_BYTES} bytes",
            )
        try:
            return _loads(raw)
        except ValueError:
            raise ServedError("malformed", "the server's answer is not JSON") from None


def open_wire(address: str) -> ModelWire:
    """The wire to ``address``: ``HttpWire``, and the seam a test replaces (``FakeModelServer.install``).

    Raises:
        ModelError: ``address`` is not ``http://host:port`` (the message does not hold it).
    """
    return HttpWire(address)


# --- the check --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServedCheck:
    """What the check found; ``ok`` says whether a call can be expected to work.

    ``kind`` is ``None`` when ``ok``, else a ``ServedError`` kind or ``not_served``
    (the server answered and does not serve the card's name). ``device`` is what
    the server said of the model, when it answered.
    """

    ok: bool
    kind: str | None
    message: str
    device: str | None


def _echo(text: str) -> str:
    """Something the server said, to be repeated in a message: printable, short."""
    return "".join(c if c.isprintable() else " " for c in text)[:_ECHO_CHARS]


def _identity_differs(expected: str, got: str) -> str:
    """Which half of a model's identity the server's differs in: the weights, the pre-processing, or both.

    An identity is the weights' SHA-256 and the pre-processing's digest joined by
    ``+``; telling the halves apart says which cure it is: other weights on the
    server, or another checkout's rule.
    """
    expected_weights, _, expected_rule = expected.partition("+")
    got_weights, _, got_rule = got.partition("+")
    weights = expected_weights != got_weights
    rule = expected_rule != got_rule
    if weights and rule:
        return "the weights and the pre-processing differ"
    if weights:
        return "the weights differ"
    if rule:
        return "the pre-processing differs"
    return "the identity differs in form"


def _short_identity(identity: str) -> str:
    """The first characters of each half of an identity: readable, and enough to tell two apart."""
    weights, _, rule = identity.partition("+")
    return f"{_echo(weights)[:_HALF_CHARS]}+{_echo(rule)[:_HALF_CHARS]}"


def check_served(
    wire: ModelWire, card: ModelCard, *, timeout_s: float = CHECK_TIMEOUT_S
) -> ServedCheck:
    """Ask the server what it serves, and say whether it serves this card's model: a finding, never an exception.

    The server's identity for the card's name is compared with the one worked out
    from the card (the weights' SHA-256 and the rule the card names). A server that
    is not there, is slow, serves other weights or is another version of this code
    is told here, at the start of the stage, and the check **does not refuse**: it is
    reported, and every call checks again by its own answer (§8.5).

    Args:
        wire: The way to the server.
        card: The model asked for.
        timeout_s: The read timeout of the listing.
    """
    try:
        result = wire.post("/models/process", {}, read_timeout_s=timeout_s)
        listed = decode_listing(result_text(result, "listing"))
    except ServedError as exc:
        return ServedCheck(False, exc.kind, f"model server: {exc}", None)
    found = next((m for m in listed if m.name == card.name), None)
    if found is None:
        names = ", ".join(_echo(m.name) for m in listed)
        serves = f"it serves: {names}" if listed else "it serves no model"
        return ServedCheck(
            False,
            "not_served",
            f"model server: the server does not serve {card.name} ({serves})",
            None,
        )
    device = _echo(found.device)
    expected = model_identity(card.weights_sha256, card.preprocessing)
    if found.identity != expected:
        return ServedCheck(
            False,
            "identity",
            f"model server: {card.name} is served with another identity than the "
            f"card's: {_identity_differs(expected, found.identity)}",
            device,
        )
    return ServedCheck(
        True,
        None,
        f"model server: {card.name} is served on {device}, identity "
        f"{_short_identity(expected)}",
        device,
    )


# --- the detector -----------------------------------------------------------------


class ServedDetector:
    """The ``Detector`` protocol over a model server: the frame goes out, the detections come back.

    What is sent is the frame's pixels and its pixel size, and the model's name: three
    keys and nothing else (never a path, a run document, a host, an account, or the
    score threshold). The **score threshold stays here**: the server keeps every
    detection and the client keeps ``score >= threshold``, the predicate of
    ``detections_from_output``, so that a served run and a local one on the same
    weights give the same detections.

    A failed call is an exception and never an empty list: "no cell" and "the server
    did not answer" are different findings. Nothing is retried, and there is no
    fallback to a model on this PC.

    ``classes`` are the card's. ``device`` is the last answer's, else the check's,
    else ``unknown``, and never ``auto``; it is a plain attribute rebound after each
    answer, because a read-only property does not satisfy the protocol's settable
    ``device``. ``check`` is the ``ServedCheck`` made when the model was loaded.
    """

    classes: tuple[str, ...]
    device: str
    check: ServedCheck

    def __init__(
        self,
        wire: ModelWire,
        *,
        model: str,
        identity: str,
        classes: tuple[str, ...],
        pixel_size_um: float,
        score_threshold: float,
        timeout_s: float,
        check: ServedCheck,
    ) -> None:
        self._wire = wire
        self._model = model
        self._identity = identity
        self._pixel_size_um = pixel_size_um
        self._score_threshold = score_threshold
        self._timeout_s = timeout_s
        self.classes = classes
        self.check = check
        self.device = check.device or "unknown"

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The detections of one 2-D frame whose score is at least the threshold.

        Raises:
            ValueError: ``frame`` is not a 2-D array with at least one pixel; nothing
                is sent.
            ServedError: The call failed (``unreachable``, ``timeout``, ``error``,
                ``malformed``), or the answer is for another model or carries another
                identity (``identity``).
        """
        array = np.asarray(frame)
        if array.ndim != 2 or array.size == 0:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        body: dict[str, object] = {
            "model": self._model,
            "image": encode_frame(array),
            "pixel_size_um": self._pixel_size_um,
        }
        try:
            result = self._wire.post(
                "/detect/process", body, read_timeout_s=self._timeout_s
            )
        except ServedError as exc:
            if exc.kind == "timeout":
                raise ServedError(
                    "timeout", f"{exc}; {TIMEOUT_ENV} sets the limit of a detect call"
                ) from None
            raise
        answer = decode_answer(result_text(result, "answer"))
        if answer.model != self._model:
            raise ServedError(
                "identity",
                f"the server answered for the model {_echo(answer.model)!r}, and "
                f"{self._model!r} was asked for",
            )
        if answer.identity != self._identity:
            raise ServedError(
                "identity",
                "the server answered with another model identity: "
                f"{_identity_differs(self._identity, answer.identity)}; it is not "
                "running the card's model (its weights, or this checkout's "
                "pre-processing)",
            )
        self.device = _echo(answer.device)
        return [d for d in answer.detections if d.score >= self._score_threshold]


# --- load_model's second place ----------------------------------------------------


def _detect_timeout_s() -> float:
    """The read timeout of a detect call: ``SMC_MODEL_TIMEOUT_S``, or ``DETECT_TIMEOUT_S``; blank is unset (FM-78)."""
    text = os.environ.get(TIMEOUT_ENV, "").strip()
    if not text:
        return DETECT_TIMEOUT_S
    refusal = f"{TIMEOUT_ENV} must be a number of seconds above 0"
    try:
        seconds = float(text)
    except ValueError:
        raise ModelError(refusal) from None
    if not (math.isfinite(seconds) and seconds > 0.0):
        raise ModelError(refusal)
    if seconds > _MAX_TIMEOUT_S:
        # `sock.settimeout(1e10)` raises OverflowError on a call, with a text that
        # names no variable: refused here, with the one that does.
        raise ModelError(f"{TIMEOUT_ENV} is longer than a day ({_MAX_TIMEOUT_S:g} s)")
    return seconds


def load_served(
    card: ModelCard,
    *,
    preprocessing: ModelPreprocessing,
    frame_um_per_px: float,
    device: str,
    score_threshold: float,
    weights: Path | None,
    address: str,
) -> ResolvedModel:
    """The model a card names, running on the server at ``address``: what ``load_model`` returns with ``SMC_MODEL_SERVER`` set.

    Refuses, before any call, what a served model cannot honour: a weights file on
    this PC, a device (the server chooses its own), a pre-processing other than the
    card's (the server runs the card's), a pixel size that is negative or not finite,
    an address that is not ``http://host:port``, a bad ``SMC_MODEL_TIMEOUT_S``. Then
    it checks the server and **reports** what it found (INFO, WARNING when the check
    is not ok): a failed check does not refuse, because a scan that goes on without
    detection is the caller's decision and ``cells`` ends ``failed`` at its first
    frame with the check's own diagnosis.

    Raises:
        ModelError: One of the refusals above; each message says what to remove.
    """
    if weights is not None:
        raise ModelError(
            "model_path names a file on this PC, and the model runs on the server: "
            f"remove model_path, or unset {SERVER_ENV} to run it here"
        )
    if device != "auto":
        raise ModelError(
            f"device={device} is for a model that runs on this PC; the server "
            f"chooses its own: remove device, or unset {SERVER_ENV}"
        )
    if preprocessing != card.preprocessing:
        raise ModelError(
            "the server runs the card's pre-processing, and low_percentile, "
            "high_percentile or model_um_per_px change it: remove them, or unset "
            f"{SERVER_ENV} to run the model here"
        )
    if not (math.isfinite(frame_um_per_px) and frame_um_per_px >= 0.0):
        raise ModelError(
            "the frames' pixel size must be a number of at least 0 um (0 is "
            f"unknown), got {frame_um_per_px}"
        )
    wire = open_wire(address)
    timeout_s = _detect_timeout_s()
    identity = model_identity(card.weights_sha256, card.preprocessing)
    check = check_served(wire, card)
    if check.ok:
        log.info("%s", check.message)
    else:
        log.warning("%s", check.message)
    return ResolvedModel(
        name=card.name,
        identity=identity,
        weights_sha256=card.weights_sha256,
        runs_on="served",
        detector=ServedDetector(
            wire,
            model=card.name,
            identity=identity,
            classes=card.classes,
            pixel_size_um=frame_um_per_px,
            score_threshold=score_threshold,
            timeout_s=timeout_s,
            check=check,
        ),
    )
