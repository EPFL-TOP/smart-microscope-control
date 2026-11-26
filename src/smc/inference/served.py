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
import math
import socket
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, Protocol
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt
import tifffile

from smc.inference.detector import Detection
from smc.inference.errors import ModelError

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
    "ServedError",
    "ServedKind",
    "ServedModel",
    "decode_answer",
    "decode_frame",
    "decode_listing",
    "encode_answer",
    "encode_frame",
    "encode_listing",
    "error_from_status",
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
#: The version of the two documents below; a server of another one is "malformed".
SCHEMA = 1

#: The server's own text is kept to a line of this many characters.
_SERVER_TEXT_CHARS = 200
_BAD_ADDRESS = f"{SERVER_ENV} is not an address like http://host:8000"

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


def unreachable(reason: str) -> ServedError:
    """The server cannot be reached: ``reason`` is the stack's own words, never the address."""
    return ServedError(
        "unreachable",
        f"the model server does not answer ({reason}); is smc serve-models "
        f"running there, and is {SERVER_ENV} its address?",
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
    number = float(value)
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

    Raises:
        ModelError: ``address`` is not of that shape.
    """
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
            raise unreachable(_os_reason(exc)) from None
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
            raise unreachable("the server closed the connection") from None
        except (OSError, http.client.HTTPException):
            raise unreachable("the connection broke") from None
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
