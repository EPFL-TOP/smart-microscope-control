"""``HttpWire`` on a real loopback socket: what a call sends, and what each way it can fail says (inference.md §8.5, §8.9).

The server is the standard library's ``ThreadingHTTPServer`` in a thread, answering as
the kit does (201 and a list, 500 ``{"message"}``, 404 and 422 ``{"detail"}``). It runs in
the light set, so every handler that waits waits on a ``threading.Event`` the test sets,
and the server's threads are daemons: ``server_close()`` joins the handler threads that
are not, and a handler that waits on an event would deadlock the teardown (measured).
No test asserts on a duration (FM-44): a timeout is a handler that does not answer and a
read timeout of 0.3 s, and what is asserted is the kind.
"""

from __future__ import annotations

import json
import socket
import threading
import traceback
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from smc.inference import served
from smc.inference.errors import ModelError
from smc.inference.served import (
    HttpWire,
    ServedError,
    decode_frame,
    encode_frame,
    open_wire,
)

NO_ADDRESS = "SMC_MODEL_SERVER is not an address like http://host:8000"
LISTING_TEXT = json.dumps({"schema": 1, "models": []})
OK_BODY = json.dumps(
    [{"type": "text", "data": LISTING_TEXT, "data_params": {"name": "models"}}]
).encode("ascii")


class _Handler(BaseHTTPRequestHandler):
    """Records the request on the server object, then lets the test's ``reply`` answer it."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        server: _Server = self.server  # type: ignore[assignment]
        server.requests.append((self.path, body))
        server.reply(self)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _Server(ThreadingHTTPServer):
    """A loopback server whose state is its own: a class attribute would be shared (RUF012)."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[tuple[str, bytes]] = []
        self.release = threading.Event()
        self.reply: Callable[[BaseHTTPRequestHandler], None] = _answer(201, OK_BODY)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up (a timeout test) leaves the handler writing to a
        # closed socket; the traceback on stderr is noise.
        pass


def _answer(status: int, body: bytes) -> Callable[[BaseHTTPRequestHandler], None]:
    def reply(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    return reply


@pytest.fixture
def server() -> Iterator[_Server]:
    """A loopback server on a free port; the event is set before the teardown (FM-44)."""
    httpd = _Server()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(10)
        assert not thread.is_alive(), "the test server did not stop"


def _wire(server: _Server) -> HttpWire:
    host, port = server.server_address[:2]
    return HttpWire(f"http://{host}:{port}")


def _closed_port() -> int:
    """A port that was bound and is not any more: nothing listens there."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _post(wire: HttpWire, *, read_timeout_s: float = 5.0) -> object:
    return wire.post("/models/process", {}, read_timeout_s=read_timeout_s)


def _error(wire: HttpWire, **kwargs: Any) -> ServedError:
    with pytest.raises(ServedError) as caught:
        _post(wire, **kwargs)
    return caught.value


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_a_201_answer_is_parsed_and_the_request_carries_exactly_three_keys(
    server: _Server, dtype: Any
) -> None:
    frame = (np.arange(12 * 20).reshape(12, 20) * 7).astype(dtype)
    body = {"model": "m", "image": encode_frame(frame), "pixel_size_um": 0.275}

    answer = _wire(server).post("/detect/process", body, read_timeout_s=5.0)

    assert answer == json.loads(OK_BODY)
    (path, raw), *others = server.requests
    assert not others
    assert path == "/detect/process"
    sent = json.loads(raw)
    assert set(sent) == {"model", "image", "pixel_size_um"}
    assert sent["model"] == "m"
    assert sent["pixel_size_um"] == 0.275
    got = decode_frame(sent["image"])
    assert got.dtype == frame.dtype
    assert np.array_equal(got, frame)


def test_a_closed_port_is_unreachable_and_the_message_holds_no_address() -> None:
    port = _closed_port()

    error = _error(HttpWire(f"http://127.0.0.1:{port}"))

    assert error.kind == "unreachable"
    shown = "".join(traceback.format_exception(error))
    assert "127.0.0.1" not in shown
    assert str(port) not in shown


def test_a_server_that_does_not_answer_is_a_timeout(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The connect timeout is raised well above the handler's own bound, so that
    # only `settimeout(read_timeout_s)` ends the wait: with the read timeout left
    # at the connect one, the call would run until the handler answers 201 below.
    monkeypatch.setattr(served, "CONNECT_TIMEOUT_S", 30.0)

    def silent_then_answer(handler: BaseHTTPRequestHandler) -> None:
        server.release.wait(5.0)
        _answer(201, OK_BODY)(handler)

    server.reply = silent_then_answer

    error = _error(_wire(server), read_timeout_s=0.3)
    server.release.set()

    assert error.kind == "timeout"
    assert "0.3" in str(error)


def test_a_server_that_closes_the_connection_is_unreachable(server: _Server) -> None:
    # A handler that returns without writing: HTTP/1.0 closes the connection.
    server.reply = lambda handler: None

    assert _error(_wire(server)).kind == "unreachable"


@pytest.mark.parametrize(
    ("status", "payload", "said"),
    [
        (
            500,
            {
                "error_type": "internal_server_error",
                "message": "CUDA out of memory" + " x" * 150 + "\nsecond line SECRET",
            },
            "CUDA out of memory",
        ),
        (404, {"detail": "Algorithm detect not found"}, "Algorithm detect not found"),
        (
            422,
            {
                "detail": [
                    {
                        "type": "extra_forbidden",
                        "loc": ["body", "threshold"],
                        "msg": "Extra inputs are not permitted",
                        "input": "FRAMEPIXELS" * 50,
                    }
                ]
            },
            "body.threshold: Extra inputs are not permitted",
        ),
    ],
    ids=["500", "404", "422"],
)
def test_an_error_status_is_an_error_with_the_servers_first_line(
    server: _Server, status: int, payload: dict[str, Any], said: str
) -> None:
    server.reply = _answer(status, json.dumps(payload).encode("ascii"))

    error = _error(_wire(server))

    text = str(error)
    assert error.kind == "error"
    assert f"HTTP {status}" in text
    assert "the server said: " + said in text
    assert "SECRET" not in text
    assert "FRAMEPIXELS" not in text
    # A line of at most 200 characters: the 500's message is longer than that.
    assert len(text.split("the server said: ", 1)[1]) <= 200


def test_an_answer_that_is_not_json_is_malformed(server: _Server) -> None:
    server.reply = _answer(201, b"<html>not the kit</html>")

    assert _error(_wire(server)).kind == "malformed"


def test_an_answer_with_a_nan_is_malformed(server: _Server) -> None:
    # Python's `json` reads `NaN`, which is not JSON; a number that overflows to
    # infinity is valid JSON and is the codecs' to refuse (test_inference_served.py).
    server.reply = _answer(201, b"[NaN]")

    assert _error(_wire(server)).kind == "malformed"


def test_an_answer_over_the_limit_is_malformed(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(served, "MAX_ANSWER_BYTES", 1000)
    server.reply = _answer(201, json.dumps(["x" * 5000]).encode("ascii"))

    error = _error(_wire(server))

    assert error.kind == "malformed"
    assert "1000" in str(error)


def test_a_proxy_in_the_environment_does_not_divert_the_call(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The proxy points at a closed port, not at the server: a client that used it
    # would fail, and one that ignored it reaches the server (FM-152).
    dead = f"http://127.0.0.1:{_closed_port()}"
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.setenv(name, dead)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    assert _post(_wire(server)) == json.loads(OK_BODY)
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "address",
    [
        "",
        "localhost:8000",
        "ftp://h:1",
        "http://h",
        "http://u:p@h:1",
        "http://h:1/x",
        "http://h:99999",
        "http://h:0",
        "http://h:1?q=1",
        "http://[::1:1",
    ],
)
def test_an_address_that_is_not_http_host_port_is_refused_without_echoing_it(
    address: str,
) -> None:
    with pytest.raises(ModelError) as caught:
        open_wire(address)

    assert str(caught.value) == NO_ADDRESS
    # Neither the value nor, on a port that is out of range, the exception it hid.
    shown = "".join(traceback.format_exception(caught.value))
    for part in ("99999", "u:p", "localhost:8000", "ftp"):
        assert part not in shown


@pytest.mark.parametrize(
    "address", ["http://h:8000", "https://h:8000", "http://[::1]:8000", "http://h:1/"]
)
def test_the_addresses_an_operator_writes_are_accepted(address: str) -> None:
    assert isinstance(open_wire(address), HttpWire)


def test_the_wires_repr_does_not_show_the_address() -> None:
    wire = HttpWire("http://lab-gpu-7:48151")

    assert "lab-gpu-7" not in repr(wire)
    assert "48151" not in repr(wire)
