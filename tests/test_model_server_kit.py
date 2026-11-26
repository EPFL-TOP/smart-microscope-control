"""A real model server: the kit's app behind uvicorn on a loopback port, and the client against it (inference.md §8.9).

Skipped without ``imaging-server-kit`` and ``uvicorn``, which are the ``serve`` extra and
in no CI job (the kit's wheels on Windows with Python 3.14 are not measured): it is run by
hand, in a venv with ``pip install imaging-server-kit==0.0.12`` and ``-e .``, and the
report of the issue says so. What it pins is the wire as the **real** server speaks it,
which the fake model server only imitates: the route list, the JSON of a 500 and of a 422,
the image a function receives, and a clean start and stop with no algorithm hub.

The server is uvicorn in a thread on a socket the test bound and listened on, so the port
is known before it starts and a connection waits in the backlog until it serves; it is
stopped with ``should_exit`` and the thread is joined with a bound and asserted gone
(FM-44). Uvicorn prints a traceback on stderr for every 500 (measured): no test asserts on
the output, only on what the client sees.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("imaging_server_kit", reason="the serve extra is not installed")
uvicorn = pytest.importorskip("uvicorn", reason="the serve extra is not installed")

from smc.inference.cards import ModelCard  # noqa: E402
from smc.inference.detector import Detection  # noqa: E402
from smc.inference.models import load_model, model_identity  # noqa: E402
from smc.inference.preprocessing import ModelPreprocessing  # noqa: E402
from smc.inference.served import (  # noqa: E402
    SERVER_ENV,
    HttpWire,
    ServedDetector,
    ServedError,
    check_served,
    encode_frame,
)
from smc.inference.server import ModelServer, build_app  # noqa: E402
from smc.testing import StubDetector, stub_model  # noqa: E402

NATIVE = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
SHA = "a" * 64
CARD = ModelCard(
    name="kit-stub",
    weights="w.pth",
    weights_sha256=SHA,
    classes=("single", "doublet", "debris"),
    trained_um_per_px=None,
    preprocessing=NATIVE,
)
FOUND = Detection(10.0, 20.0, 30.0, 40.0, "single", float(np.float32(0.93)))


@dataclass
class _Running:
    """A uvicorn server in a thread: its port, and what is needed to stop it."""

    port: int
    server: Any
    thread: threading.Thread
    sock: socket.socket

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(30)
        assert not self.thread.is_alive(), "the server did not stop"
        self.sock.close()


class _Recording(ModelServer):
    """A ``ModelServer`` that notes what the kit handed it: the frame as it arrived and the pixel size."""

    def __init__(self, models: Any) -> None:
        super().__init__(models)
        self.received: list[tuple[str, Any, tuple[int, ...], int, float]] = []

    def detect(self, model: str, frame: npt.NDArray[Any], pixel_size_um: float) -> str:
        self.received.append(
            (model, frame.dtype, frame.shape, int(frame[0, 0]), pixel_size_um)
        )
        return super().detect(model, frame, pixel_size_um)


@pytest.fixture
def start() -> Iterator[Callable[[ModelServer], _Running]]:
    """Start the real app for a ``ModelServer``; every server it started is stopped, bounded."""
    running: list[_Running] = []

    def run(model_server: ModelServer) -> _Running:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        server = uvicorn.Server(
            uvicorn.Config(build_app(model_server), log_level="warning")
        )
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        started = _Running(int(sock.getsockname()[1]), server, thread, sock)
        running.append(started)
        return started

    try:
        yield run
    finally:
        for started in running:
            if started.thread.is_alive():
                started.stop()


def _stub_server(stub: StubDetector) -> _Recording:
    model = stub_model(stub, name=CARD.name, weights_sha256=SHA, preprocessing=NATIVE)
    return _Recording({CARD.name: model})


def _wire(started: _Running) -> HttpWire:
    return HttpWire(f"http://127.0.0.1:{started.port}")


def _body(key: int = 7, **extra: object) -> dict[str, object]:
    return {
        "model": CARD.name,
        "image": encode_frame(np.full((60, 80), key, dtype=np.uint16)),
        "pixel_size_um": 0.275,
        **extra,
    }


def test_the_services_route_lists_models_and_detect(
    start: Callable[[ModelServer], _Running],
) -> None:
    started = start(_stub_server(StubDetector()))
    connection = http.client.HTTPConnection("127.0.0.1", started.port, timeout=30)
    try:
        connection.request("GET", "/services")
        response = connection.getresponse()
        listed = json.loads(response.read())
    finally:
        connection.close()

    assert response.status == 200
    assert listed == {"services": ["models", "detect"]}


def test_the_real_server_lists_its_models_and_answers_a_frame(
    start: Callable[[ModelServer], _Running], monkeypatch: pytest.MonkeyPatch
) -> None:
    model_server = _stub_server(StubDetector({7: [FOUND]}, device="cuda"))
    started = start(model_server)
    monkeypatch.setenv(SERVER_ENV, f"http://127.0.0.1:{started.port}")

    # The way `cells` gets a model: `load_model` with the variable set.
    resolved = load_model(
        CARD,
        preprocessing=NATIVE,
        frame_shape_px=(60, 80),
        frame_um_per_px=0.275,
        device="auto",
        score_threshold=0.5,
    )
    detector = resolved.detector
    assert isinstance(detector, ServedDetector)
    assert detector.check.ok, detector.check.message
    assert detector.device == "cuda"
    found = detector.detect(np.full((60, 80), 7, dtype=np.uint16))

    assert resolved.runs_on == "served"
    assert found == [FOUND]
    # What the function received through the kit: the frame as `uint16`, its first
    # pixel (which the stub keys on), the pixel size.
    ((name, dtype, shape, first_pixel, pixel_size),) = model_server.received
    assert (name, dtype, shape, first_pixel) == (CARD.name, np.uint16, (60, 80), 7)
    assert pixel_size == 0.275
    weights, rule = model_identity(SHA, NATIVE).split("+")
    assert detector.check.message.endswith(f"{weights[:12]}+{rule[:12]}")


def test_the_check_of_a_real_server_that_does_not_serve_the_model(
    start: Callable[[ModelServer], _Running],
) -> None:
    other = stub_model(StubDetector(), name="another")
    started = start(_Recording({"another": other}))

    found = check_served(_wire(started), CARD)

    assert (found.ok, found.kind) == (False, "not_served")
    assert "does not serve kit-stub (it serves: another)" in found.message


def test_a_function_that_raises_is_an_error_with_its_message(
    start: Callable[[ModelServer], _Running],
) -> None:
    def fall_over(key: int) -> None:
        raise RuntimeError("out of memory on the device\nsecond line")

    started = start(_stub_server(StubDetector(on_frame=fall_over)))

    with pytest.raises(ServedError) as caught:
        _wire(started).post("/detect/process", _body(), read_timeout_s=30.0)

    assert caught.value.kind == "error"
    assert "HTTP 500" in str(caught.value)
    assert str(caught.value).endswith("the server said: out of memory on the device")


def test_an_unknown_model_is_an_error(
    start: Callable[[ModelServer], _Running],
) -> None:
    started = start(_stub_server(StubDetector()))

    with pytest.raises(ServedError) as caught:
        _wire(started).post(
            "/detect/process", {**_body(), "model": "nope"}, read_timeout_s=30.0
        )

    assert caught.value.kind == "error"
    assert "does not serve 'nope'" in str(caught.value)
    assert "it serves: kit-stub" in str(caught.value)


def test_a_fourth_key_and_a_negative_pixel_size_are_a_422(
    start: Callable[[ModelServer], _Running],
) -> None:
    # The claim of the wire: the kit's parameter model is `extra="forbid"`, so a client
    # that sent the threshold or a path would be refused, and the pixel size has `ge=0`.
    started = start(_stub_server(StubDetector()))
    wire = _wire(started)

    with pytest.raises(ServedError) as extra:
        wire.post("/detect/process", _body(score_threshold=0.5), read_timeout_s=30.0)
    with pytest.raises(ServedError) as negative:
        wire.post(
            "/detect/process", {**_body(), "pixel_size_um": -1.0}, read_timeout_s=30.0
        )

    assert "HTTP 422" in str(extra.value)
    assert "Extra inputs are not permitted" in str(extra.value)
    assert "HTTP 422" in str(negative.value)


def test_a_slow_function_is_a_timeout(
    start: Callable[[ModelServer], _Running],
) -> None:
    release = threading.Event()

    def hold(key: int) -> None:
        release.wait(30)

    started = start(_stub_server(StubDetector(on_frame=hold)))
    try:
        with pytest.raises(ServedError) as caught:
            _wire(started).post("/detect/process", _body(), read_timeout_s=0.5)
    finally:
        # Before the teardown: the kit's worker thread is waiting on it.
        release.set()

    assert caught.value.kind == "timeout"


def test_the_server_starts_and_stops_with_no_hub(
    start: Callable[[ModelServer], _Running],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The kit's own lifespan asks an "algorithm hub" for a host that does not exist and
    # deregisters unguarded at shutdown; ours does neither (inference.md §8.1).
    started = start(_stub_server(StubDetector()))
    assert check_served(_wire(started), CARD).ok

    started.stop()

    seen = capsys.readouterr()
    assert "hub" not in seen.out.lower()
    assert "shutdown failed" not in caplog.text
    assert "startup failed" not in caplog.text
    assert not started.thread.is_alive()
