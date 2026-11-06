"""A Ctrl-C of ``smc ui`` with a session open prints one line, not a traceback (m5 §11.5, #332).

A stop of Bokeh's server discards the sessions and detaches each connection from its
session before the socket's own close reaches ``on_close``, which then reads
``connection.session``: an ``AssertionError`` that Tornado logs as ``Uncaught
exception GET /ws`` on every Ctrl-C with the page open in a browser. ``serve()``
drops that one record while it stops the server.

The end-to-end test goes through a real server and a real session, interrupted by
``_thread.interrupt_main`` (FM-161): the case a fake thread cannot show. Every wait
is bounded and every join is checked (FM-44). Measured on Bokeh 3.10.1, Tornado 6.6
and Python 3.13: ``serve()`` returns about a second after the interrupt, and the
``tornado.application`` logger holds exactly one record, an ``AssertionError``.
"""

from __future__ import annotations

import _thread
import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel  # noqa: F401  # CI must fail, not skip, when the extra is missing
else:
    pytest.importorskip("panel")

import smc.ui.plate_page as plate_page_module
from smc.hardware import Microscope
from smc.plugins.registry import GROUP
from smc.ui.controller import PlatePage
from smc.ui.plate_page import (
    _quiet_session_close,
    _ServerStopFilter,
    serve,
)

TORNADO = "tornado.application"
CLOSING = "Uncaught exception GET /ws (::1)"

# `interrupt_main()` interrupts the main thread: a test that is not in it would end
# the wrong thread (the same rule as `tests/unit/test_executor.py`).
main_thread_only = pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="interrupt_main() interrupts the main thread",
)

#: How long the helper thread waits for the server and for a session, in seconds; a
#: watchdog interrupts the main thread after the same bound, so a test that hangs ends.
WAIT_S = 30.0


class _Collect(logging.Handler):
    """What ``tornado.application`` lets through its filters, as (message, exception class)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.seen: list[tuple[str, type[BaseException] | None]] = []

    def emit(self, record: logging.LogRecord) -> None:
        error = None if record.exc_info is None else record.exc_info[0]
        self.seen.append((record.getMessage(), error))


@pytest.fixture
def collected() -> Iterator[_Collect]:
    handler = _Collect()
    logger = logging.getLogger(TORNADO)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _record(message: str, error: BaseException | None) -> logging.LogRecord:
    return logging.LogRecord(
        name=TORNADO,
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None if error is None else (type(error), error, None),
    )


def _logged(message: str, error: BaseException | None) -> None:
    """What Tornado does: the exception, under the application logger."""
    logging.getLogger(TORNADO).error(
        message,
        exc_info=None if error is None else (type(error), error, None),
    )


def test_the_filter_drops_only_the_sessions_closing_assertion() -> None:
    stopping = _ServerStopFilter()
    assert stopping.dropped == 0
    # The record of the finding: the closing socket's assertion.
    assert not stopping.filter(_record(CLOSING, AssertionError()))
    assert stopping.dropped == 1
    # The same words with another exception hide a different failure: they pass.
    assert stopping.filter(_record(CLOSING, ValueError("a real handler failure")))
    # Another request's assertion passes: the message alone bounds nothing.
    assert stopping.filter(
        _record("Uncaught exception GET /static (::1)", AssertionError())
    )
    # No exception at all passes too.
    assert stopping.filter(_record(CLOSING, None))
    assert stopping.dropped == 1


def test_the_filter_counts_each_record_it_drops() -> None:
    stopping = _ServerStopFilter()
    for _ in range(3):
        assert not stopping.filter(_record(CLOSING, AssertionError()))
    assert stopping.dropped == 3


def test_the_record_after_the_filter_is_removed_is_logged(
    collected: _Collect,
) -> None:
    with _quiet_session_close() as stopping:
        _logged(CLOSING, AssertionError())
        assert collected.seen == []
        assert stopping.dropped == 1
        # While the server stops, a failure of another kind is still logged.
        _logged(CLOSING, ValueError("a real failure"))
        assert collected.seen == [(CLOSING, ValueError)]
    # An AssertionError of `GET /ws` while the server runs is a finding, not a stop.
    _logged(CLOSING, AssertionError())
    assert collected.seen == [(CLOSING, ValueError), (CLOSING, AssertionError)]
    assert stopping.dropped == 1


def test_the_filter_is_removed_when_the_stop_raises(collected: _Collect) -> None:
    logger = logging.getLogger(TORNADO)
    before = list(logger.filters)
    with (
        pytest.raises(RuntimeError, match=r"^the stop failed$"),
        _quiet_session_close(),
    ):
        raise RuntimeError("the stop failed")
    assert logger.filters == before


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


@pytest.fixture
def page(fake_microscope: Microscope, run_doc: Path) -> Iterator[PlatePage]:
    plugins = [
        EntryPoint(
            name="plate-registration",
            value="smc.plugins.plate_registration.plugin:PlateRegistrationPlugin",
            group=GROUP,
        ),
        EntryPoint(
            name="scan-plan",
            value="smc.plugins.scan_plan.plugin:ScanPlanPlugin",
            group=GROUP,
        ),
    ]
    controller = PlatePage(fake_microscope, run_doc, entry_points=plugins)
    try:
        yield controller
    finally:
        controller.close()


class _FakeServer:
    """What ``pn.serve(..., threaded=True)`` returns, as far as ``serve()`` uses it.

    The first wait is the operator's Ctrl-C. ``stop`` does what Bokeh's stop does to
    an open session: it logs the closing assertion once for each session it closes,
    and says which filters were on the logger while it did.
    """

    def __init__(self, *, closes: int) -> None:
        self.closes = closes
        self.alive = True
        self.filters_during_stop: list[type] = []
        self._interrupted = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        if not self._interrupted:
            self._interrupted = True
            raise KeyboardInterrupt

    def stop(self) -> None:
        self.filters_during_stop = [type(f) for f in logging.getLogger(TORNADO).filters]
        for _ in range(self.closes):
            _logged(CLOSING, AssertionError())
        self.alive = False


@pytest.mark.parametrize("closes", [0, 1, 2])
def test_serve_drops_what_the_stop_logs_and_counts_it(
    page: PlatePage,
    collected: _Collect,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    closes: int,
) -> None:
    # `serve()` itself, on every Bokeh version: the vendor's record is dropped while the
    # server stops, the page's log counts it in one line, and the filter is attached
    # around the stop only (m5 §11.5). The real server shows the traceback; this shows
    # the wiring, on a version that closes quietly too.
    server = _FakeServer(closes=closes)
    monkeypatch.setattr(plate_page_module.pn, "serve", lambda *a, **k: server)
    monkeypatch.setattr("smc.cli._say", lambda text: None)
    logger = logging.getLogger(TORNADO)
    before = list(logger.filters)

    with caplog.at_level(logging.INFO, logger="smc.ui"):
        serve(page, port=5006)

    assert server.filters_during_stop == [*map(type, before), _ServerStopFilter]
    assert logger.filters == before
    assert collected.seen == []
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "smc.ui" and "the server's stop closed" in r.getMessage()
    ]
    assert lines == (
        [f"the server's stop closed {closes} open sessions"] if closes else []
    )
    # The same record, while the server runs, is a finding and is logged.
    _logged(CLOSING, AssertionError())
    assert collected.seen == [(CLOSING, AssertionError)]


@main_thread_only
def test_a_ctrl_c_with_a_session_open_prints_no_traceback(
    page: PlatePage,
    collected: _Collect,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from bokeh.client import pull_session

    port = _free_port()
    announced = threading.Event()
    problems: list[str] = []
    said: list[str] = []

    def say(text: str) -> None:
        # `serve()` says this after `pn.serve(threaded=True)` returned and before its
        # waiting loop: the interrupt cannot come earlier. The server may not accept
        # connections yet (Panel 1.9 does not), which the helper waits for. What it
        # says is kept: a stop that did not end says so here, and nowhere else.
        said.append(text)
        announced.set()

    monkeypatch.setattr("smc.cli._say", say)
    session_open = threading.Event()
    interrupted = threading.Event()
    done = threading.Event()
    # Set once `serve()` has returned or raised: no interrupt may follow it. A late one
    # lands in pytest, not in the test, and aborts the session with the failure of this
    # test unreported (the helper waits up to WAIT_S for a server that never answers).
    ended = threading.Event()

    def operator() -> None:
        """What a person does: opens the page in a browser, then presses Ctrl-C."""
        session = None
        try:
            if not announced.wait(WAIT_S):
                problems.append("the server was not announced")
                return
            deadline = time.monotonic() + WAIT_S
            while session is None and not ended.is_set():
                try:
                    session = pull_session(url=f"http://localhost:{port}/")
                except OSError:  # the server is not accepting connections yet
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            if session is not None:
                session_open.set()
        except Exception as exc:  # the test says it, below
            problems.append(f"{type(exc).__name__}: {exc}")
        finally:
            # Only while `serve()` still waits for it.
            if not ended.is_set():
                interrupted.set()
                _thread.interrupt_main()
        # The session stays open while the server stops: a closed one would hide it.
        done.wait(WAIT_S)
        if session is not None:
            # The server is gone by now: a failure to close says nothing about it.
            with contextlib.suppress(Exception):
                session.close()

    helper = threading.Thread(target=operator, name="smc-test-operator", daemon=True)
    # If the helper never interrupts, this ends the test instead of hanging the suite.
    watchdog = threading.Timer(WAIT_S * 2, _thread.interrupt_main)
    watchdog.daemon = True
    helper.start()
    watchdog.start()
    started = time.monotonic()
    try:
        with caplog.at_level(logging.INFO, logger="smc.ui"):
            serve(page, port=port)
    finally:
        ended.set()
        watchdog.cancel()
        done.set()
        helper.join(WAIT_S)
    assert not helper.is_alive()
    assert problems == []
    assert session_open.is_set()
    assert interrupted.is_set()
    # `serve()` stops the server by itself, within its own bound, and says so when it
    # could not: the server thread is not left running (Panel's is not a daemon: it
    # would hold the process, and the test runner, at exit).
    assert time.monotonic() - started < WAIT_S
    assert [text for text in said if "did not stop" in text] == []
    assert [
        t.name
        for t in threading.enumerate()
        if type(t).__name__ == "StoppableThread" and t.is_alive()
    ] == []

    # What a person would have read: no record of the closing socket's assertion. (The
    # console is not asserted: pytest takes the log records before any stream sees
    # them, so only this handler on the logger can tell.)
    assert [
        m for m, _ in collected.seen if m.startswith("Uncaught exception GET /ws")
    ] == []
    # The page's own log says what the stop did, once, in one line: that it closed the
    # one open session. Bokeh 3.9 (Panel 1.9, the floor) closes quietly, so there is no
    # record to drop and no line (measured, with a probe that stops the server with one
    # session open); Bokeh 3.10 raises the assertion, and the line is the count of it,
    # which `test_serve_drops_what_the_stop_logs_and_counts_it` asserts exactly.
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "smc.ui" and "the server's stop closed" in r.getMessage()
    ]
    assert lines in ([], ["the server's stop closed 1 open sessions"])
