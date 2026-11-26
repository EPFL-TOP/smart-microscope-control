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

import http.client
import json
import socket
import ssl
import threading
import traceback
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from smc.inference import served
from smc.inference.cards import ModelCard
from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.models import load_model
from smc.inference.preprocessing import ModelPreprocessing
from smc.inference.served import (
    HttpWire,
    ServedError,
    decode_frame,
    encode_frame,
    open_wire,
)
from smc.inference.server import ModelServer
from smc.testing import StubDetector, stub_model

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


# A space or a control character in the host: `urlsplit` accepts it and `http.client`
# raises `InvalidURL` (a text that holds the host) when it builds the connection,
# outside every `try` of the wire; so it is refused with the others, by the one fixed
# message (FM-50, FM-277).
HOST_WITH_A_SPACE_OR_A_CONTROL_CHARACTER = [
    "http://10.0.0.5 :8000",
    "http://my host:8000",
    "http://lab-gpu\x07:8000",
    "http://lab-gpu\x7f:8000",
    "http://lab gpu\t:8000",
]


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
        *HOST_WITH_A_SPACE_OR_A_CONTROL_CHARACTER,
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
    for part in ("99999", "u:p", "localhost:8000", "ftp", "10.0.0.5", "my host", "gpu"):
        assert part not in shown


@pytest.mark.parametrize("address", HOST_WITH_A_SPACE_OR_A_CONTROL_CHARACTER)
def test_a_host_with_a_space_in_the_variable_is_a_model_error_and_not_the_clients_exception(
    address: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The address of the variable `load_model` reads: a `ModelError` with the fixed
    # message, which `cells` ends the stage with, and not an `InvalidURL` that holds the
    # lab's host and goes past every `except ModelError` into a diagnosis the run
    # folder keeps.
    monkeypatch.setenv(served.SERVER_ENV, address)
    card = ModelCard(
        name="stub",
        weights="w.pth",
        weights_sha256="a" * 64,
        classes=("single",),
        trained_um_per_px=None,
        preprocessing=ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None),
    )

    with pytest.raises(ModelError) as caught:
        load_model(
            card,
            preprocessing=card.preprocessing,
            frame_shape_px=(64, 64),
            frame_um_per_px=0.0,
            device="auto",
            score_threshold=0.5,
        )

    assert str(caught.value) == NO_ADDRESS
    shown = "".join(traceback.format_exception(caught.value))
    for part in ("10.0.0.5", "my host", "gpu"):
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


def test_a_model_asked_for_by_name_runs_through_a_real_socket(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole path of a served model on a loopback socket: `load_model` opens an
    # `HttpWire`, checks the listing of a real `ModelServer`, and a frame goes to its
    # stub and back. Only the HTTP layer is the standard library's here.
    sha = "a" * 64
    rule = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
    card = ModelCard(
        name="socket-stub",
        weights="w.pth",
        weights_sha256=sha,
        classes=("single",),
        trained_um_per_px=None,
        preprocessing=rule,
    )
    found = Detection(1.0, 2.0, 3.0, 4.0, "single", 0.9)
    stub = StubDetector({7: [found]}, classes=("single",), device="cuda")
    model_server = ModelServer(
        {
            card.name: stub_model(
                stub, name=card.name, weights_sha256=sha, preprocessing=rule
            )
        }
    )

    def reply(handler: BaseHTTPRequestHandler) -> None:
        path, raw = server.requests[-1]
        if path == "/models/process":
            text, name = model_server.listing(), "models"
        else:
            sent = json.loads(raw)
            text = model_server.detect(
                sent["model"], decode_frame(sent["image"]), sent["pixel_size_um"]
            )
            name = "detections"
        result = [{"type": "text", "data": text, "data_params": {"name": name}}]
        _answer(201, json.dumps(result).encode("ascii"))(handler)

    server.reply = reply
    host, port = server.server_address[:2]
    monkeypatch.setenv(served.SERVER_ENV, f"http://{host}:{port}")

    resolved = load_model(
        card,
        preprocessing=rule,
        frame_shape_px=(6, 9),
        frame_um_per_px=0.275,
        device="auto",
        score_threshold=0.5,
    )
    detections = resolved.detector.detect(np.full((6, 9), 7, dtype=np.uint16))

    assert resolved.runs_on == "served"
    assert detections == [found]
    assert resolved.detector.device == "cuda"
    check_path, detect_path = [path for path, _ in server.requests]
    assert (check_path, detect_path) == ("/models/process", "/detect/process")
    # On the wire, not only on the fake: the keys of what a detect call sends.
    assert set(json.loads(server.requests[0][1])) == set()
    assert set(json.loads(server.requests[1][1])) == {"model", "image", "pixel_size_um"}


# --- the connection, with `http.client` replaced: what no loopback server can show ---


def _stub_connections(
    monkeypatch: pytest.MonkeyPatch,
    *,
    connect: BaseException | None = None,
    request: BaseException | None = None,
) -> list[dict[str, Any]]:
    """Replace both connection classes by one that records how it was built and fails as told.

    A refused loopback connection never reaches the connect timeout, and a plain loopback
    server cannot speak TLS: what is checked here is how the wire builds a connection (the
    class for the scheme, the timeout of the connect) and what it says of each failure.
    """
    built: list[dict[str, Any]] = []

    def connection(tls: bool) -> type:
        class Connection:
            sock = None

            def __init__(self, host: str, port: int, timeout: float) -> None:
                built.append(
                    {"tls": tls, "host": host, "port": port, "timeout": timeout}
                )

            def connect(self) -> None:
                if connect is not None:
                    raise connect

            def request(self, *args: object, **kwargs: object) -> None:
                if request is not None:
                    raise request

            def getresponse(self) -> object:
                raise AssertionError("a response was asked for")

            def close(self) -> None:
                pass

        return Connection

    monkeypatch.setattr(http.client, "HTTPConnection", connection(False))
    monkeypatch.setattr(http.client, "HTTPSConnection", connection(True))
    return built


@pytest.mark.parametrize(
    ("address", "tls"),
    [
        ("http://gpu-box.lab.test:48151", False),
        ("https://gpu-box.lab.test:48151", True),
    ],
)
def test_the_scheme_picks_the_connection_and_the_connect_has_its_own_timeout(
    monkeypatch: pytest.MonkeyPatch, address: str, tls: bool
) -> None:
    built = _stub_connections(monkeypatch, connect=TimeoutError("timed out"))

    with pytest.raises(ServedError) as caught:
        HttpWire(address).post("/models/process", {}, read_timeout_s=7.0)

    # TLS for https and never for http; the host and port as written, the connect
    # timeout and not the call's read timeout and not none (FM-31).
    assert built == [
        {
            "tls": tls,
            "host": "gpu-box.lab.test",
            "port": 48151,
            "timeout": served.CONNECT_TIMEOUT_S,
        }
    ]
    # No answer within the connect timeout is "unreachable" (nobody is there), not the
    # read timeout's "timeout".
    assert caught.value.kind == "unreachable"
    assert "no connection within 3 s" in str(caught.value)
    shown = "".join(traceback.format_exception(caught.value))
    assert "gpu-box" not in shown
    assert "48151" not in shown


def test_a_host_name_that_cannot_be_encoded_is_unreachable_and_not_a_raw_error() -> (
    None
):
    # A label of more than 63 characters is a `UnicodeError` out of `getaddrinfo`,
    # before any name is looked up: no network is touched.
    wire = HttpWire("http://" + "a" * 64 + ".example:8000")

    error = _error(wire)

    assert error.kind == "unreachable"
    assert "aaaa" not in str(error)


def test_a_tls_failure_says_what_to_check_for_tls_and_not_to_start_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = ssl.SSLCertVerificationError(
        1,
        "certificate verify failed: Hostname mismatch, not valid for 'gpu-box.lab.test'",
    )
    _stub_connections(monkeypatch, connect=refusal)

    with pytest.raises(ServedError) as caught:
        HttpWire("https://gpu-box.lab.test:48151").post(
            "/models/process", {}, read_timeout_s=5.0
        )

    text = str(caught.value)
    assert caught.value.kind == "unreachable"
    assert "the TLS handshake failed" in text
    assert "use http:// unless the server serves TLS" in text
    # The server is running: the advice for a server that is not is the wrong one (FM-263).
    assert "is smc serve-models running" not in text
    # The certificate's own text names the host.
    assert "gpu-box" not in text
    # And not through the chain either: the original is dropped, not kept as the cause.
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_a_connection_that_breaks_says_the_server_was_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_connections(monkeypatch, request=BrokenPipeError(32, "Broken pipe"))

    with pytest.raises(ServedError) as caught:
        HttpWire("http://gpu-box.lab.test:48151").post(
            "/models/process", {}, read_timeout_s=5.0
        )

    assert caught.value.kind == "unreachable"
    assert "the connection broke" in str(caught.value)
    assert "look at its console" in str(caught.value)
    assert "is smc serve-models running" not in str(caught.value)


def test_the_servers_own_text_is_printable_and_short(server: _Server) -> None:
    server.reply = _answer(
        500,
        json.dumps({"message": "\x1b[31mred\x00\x07 and more"}).encode("ascii"),
    )

    text = str(_error(_wire(server)))

    assert "\x1b" not in text
    assert "\x00" not in text
    assert "\x07" not in text
    assert text.endswith("the server said:  [31mred   and more")
