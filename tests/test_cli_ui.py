"""``smc ui plate`` (m3 §15.3): the order of its checks, its Ctrl-C and what releases the stand.

``smc.ui.plate_page`` is replaced by a stand-in module with a ``serve`` of the
test's own, so none of this needs Panel and it runs on every CI job; the page
itself is tested in ``tests/unit/test_plate_page.py``. The runner pins the width
(FM-37) and clears ``SMC_PROFILE``, so an operator's ``SMC_PROFILE=nikon-ti2`` never
makes ``pytest`` open a real stand (FM-40). Interrupts are raised from the
stand-in, never sent as a signal (FM-44).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import types
from collections.abc import Callable
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pymmcore_plus import CMMCorePlus
from typer.testing import CliRunner, Result

import smc.ui.controller as controller_module
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.testing import FakeCore
from smc.ui.controller import PageServerError, PlatePage

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PANEL = re.compile(r"[│╭╮╰╯─]")

FIX = 'install the ui extra: pip install -e ".[ui]"'


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class Slow:
    """A run that goes on until it is stopped."""

    name = provides = "slow"
    summary = "A test plugin that waits."
    Params = NoParams
    Result = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()
    started: ClassVar[threading.Event] = threading.Event()

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> NoResult:
        Slow.started.set()
        while True:
            ctx.wait_s(0.05)


@pytest.fixture(autouse=True)
def _fresh_started() -> None:
    Slow.started = threading.Event()


def _plain(result: Result) -> str:
    """The output without colour codes, box lines or runs of blanks (FM-72)."""
    text = _PANEL.sub(" ", _ANSI.sub("", result.output))
    return re.sub(r"[ \t]+", " ", text)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return int(probe.getsockname()[1])


def _stand_in(monkeypatch: pytest.MonkeyPatch, serve: Callable[..., None]) -> None:
    """Put a module with this ``serve`` where ``smc.ui.plate_page`` is looked up."""
    module = types.ModuleType("smc.ui.plate_page")
    module.__dict__["serve"] = serve
    monkeypatch.setitem(sys.modules, "smc.ui.plate_page", module)


@pytest.fixture
def no_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command opens a microscope."""

    def refuse(cls: type[Microscope], *args: object, **kwargs: object) -> Microscope:
        pytest.fail("the command opened a microscope")

    monkeypatch.setattr(Microscope, "open", classmethod(refuse))


@pytest.fixture
def fake_stand(monkeypatch: pytest.MonkeyPatch) -> list[Microscope]:
    """Open a ``FakeCore`` stand, as the profile it is given; the list holds each one opened."""
    opened: list[Microscope] = []

    def open_stand(
        cls: type[Microscope], profile: object = "demo", *, dry_run: bool = False
    ) -> Microscope:
        built = profile if isinstance(profile, Profile) else Profile.demo()
        microscope = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()), built, dry_run=dry_run
        )
        opened.append(microscope)
        return microscope

    monkeypatch.setattr(Microscope, "open", classmethod(open_stand))
    return opened


@pytest.fixture
def closes(monkeypatch: pytest.MonkeyPatch) -> list[Microscope]:
    """Every facade that was closed, in order."""
    closed: list[Microscope] = []
    real = Microscope.close

    def close(self: Microscope) -> None:
        closed.append(self)
        real(self)

    monkeypatch.setattr(Microscope, "close", close)
    return closed


def _plate(*args: str, doc: Path, port: int | None = None) -> Result:
    return runner.invoke(
        app,
        [
            "ui",
            "plate",
            "--run-doc",
            str(doc),
            "--port",
            str(port or _free_port()),
            *args,
        ],
    )


# --- the checks that need no stand --------------------------------------------------


def test_ui_plate_needs_a_run_document(no_stand: None) -> None:
    result = runner.invoke(app, ["ui", "plate"])
    assert result.exit_code == 2, result.output
    assert "--run-doc" in _plain(result)
    assert "a run needs --run-doc" in _plain(result)


@pytest.mark.parametrize("empty", ["", "   "])
def test_an_empty_run_document_path_is_refused(no_stand: None, empty: str) -> None:
    # `Path("")` is the current folder (FM-78).
    result = runner.invoke(app, ["ui", "plate", "--run-doc", empty])
    assert result.exit_code == 2, result.output
    assert "is empty; name the file to write" in _plain(result)


def test_a_port_out_of_range_is_refused(no_stand: None, tmp_path: Path) -> None:
    for port in ("0", "65536"):
        result = runner.invoke(
            app,
            ["ui", "plate", "--run-doc", str(tmp_path / "run.json"), "--port", port],
        )
        assert result.exit_code == 2, result.output


def test_the_missing_ui_extra_is_named_with_its_fix(
    monkeypatch: pytest.MonkeyPatch, no_stand: None, tmp_path: Path
) -> None:
    # `import panel` raises as it would on a PC without the extra, and the page
    # module is imported afresh, so the stand-in of another test is not found.
    monkeypatch.setitem(sys.modules, "panel", None)
    monkeypatch.delitem(sys.modules, "smc.ui.plate_page", raising=False)
    result = _plate(doc=tmp_path / "run.json")
    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    assert f"✗ {FIX}" in lines
    assert "import of panel halted; None in sys.modules" in lines


def test_a_taken_port_is_refused_before_the_stand_is_opened(
    monkeypatch: pytest.MonkeyPatch, no_stand: None, tmp_path: Path
) -> None:
    served: list[object] = []
    _stand_in(monkeypatch, lambda page, *, port: served.append(page))
    with socket.socket() as held:
        held.bind(("localhost", 0))
        port = held.getsockname()[1]
        result = _plate(doc=tmp_path / "run.json", port=port)
    assert result.exit_code == 2, result.output
    assert (
        f"✗ port {port} is in use; choose another with --port"
        in result.output.splitlines()
    )
    assert served == []


def test_an_unreadable_run_document_is_refused_and_left_alone(
    monkeypatch: pytest.MonkeyPatch, no_stand: None, tmp_path: Path
) -> None:
    served: list[object] = []
    _stand_in(monkeypatch, lambda page, *, port: served.append(page))
    doc = tmp_path / "run.json"
    doc.write_bytes(b"not json")
    result = _plate(doc=doc)
    assert result.exit_code == 1, result.output
    assert "is not valid JSON: Expecting value (line 1, column 1)" in _plain(result)
    assert doc.read_bytes() == b"not json"
    assert served == []


# --- serving ----------------------------------------------------------------------------


def test_ui_plate_hands_the_page_the_stand_and_releases_it_after(
    monkeypatch: pytest.MonkeyPatch,
    fake_stand: list[Microscope],
    closes: list[Microscope],
    tmp_path: Path,
) -> None:
    seen: list[tuple[PlatePage, int, bool]] = []

    def serve(page: PlatePage, *, port: int) -> None:
        seen.append((page, port, bool(closes)))

    _stand_in(monkeypatch, serve)
    port = _free_port()
    result = _plate(doc=tmp_path / "run.json", port=port)
    assert result.exit_code == 0, result.output
    [(page, served_port, closed_while_serving)] = seen
    assert isinstance(page, PlatePage)
    assert served_port == port
    assert not closed_while_serving
    assert closes == fake_stand
    assert not tmp_path.joinpath(
        "run.json"
    ).exists()  # the page writes nothing by itself


def test_ui_plate_dry_run_opens_the_stand_dry(
    monkeypatch: pytest.MonkeyPatch,
    fake_stand: list[Microscope],
    tmp_path: Path,
) -> None:
    dry: list[bool] = []
    _stand_in(monkeypatch, lambda page, *, port: dry.append(page._microscope.dry_run))
    assert _plate("--dry-run", doc=tmp_path / "run.json").exit_code == 0
    assert dry == [True]


@pytest.mark.demo
def test_ui_plate_serves_the_demo_and_releases_the_stand(
    monkeypatch: pytest.MonkeyPatch, closes: list[Microscope], tmp_path: Path
) -> None:
    pages: list[PlatePage] = []
    _stand_in(monkeypatch, lambda page, *, port: pages.append(page))
    result = _plate(doc=tmp_path / "run.json")
    assert result.exit_code == 0, result.output
    assert [type(page) for page in pages] == [PlatePage]
    assert len(closes) == 1


class _SpyPage(PlatePage):
    """A page that says when it was closed, and loads a slow plugin besides the installed ones."""

    closed: ClassVar[list[str]] = []
    order: ClassVar[list[str]] = []

    def __init__(
        self,
        microscope: Microscope,
        run_doc: Path,
        *,
        entry_points: Any = None,
    ) -> None:
        slow = EntryPoint(name="slow", value=f"{__name__}:Slow", group=GROUP)
        super().__init__(microscope, run_doc, entry_points=[slow])

    def stop(self) -> None:
        _SpyPage.order.append("stop")
        super().stop()

    def close(self) -> None:
        _SpyPage.closed.append("close")
        super().close()


@pytest.fixture
def spy_page(monkeypatch: pytest.MonkeyPatch) -> type[_SpyPage]:
    _SpyPage.closed = []
    _SpyPage.order = []
    monkeypatch.setattr(controller_module, "PlatePage", _SpyPage)
    return _SpyPage


@pytest.mark.parametrize(
    ("raises", "code", "line"),
    [
        pytest.param(
            RuntimeError("the server broke"),
            1,
            "✗ unexpected RuntimeError: the server broke — run it again as smc --debug … "
            "for the traceback, and file an issue",
            id="a-bug",
        ),
        pytest.param(
            PageServerError(
                "the plate page's server stopped on its own (is port 5006 taken?)"
            ),
            1,
            "✗ the plate page's server stopped on its own (is port 5006 taken?)",
            id="the-server-ended-alone",
        ),
    ],
)
def test_the_page_is_closed_whatever_ends_serve(
    monkeypatch: pytest.MonkeyPatch,
    fake_stand: list[Microscope],
    closes: list[Microscope],
    spy_page: type[_SpyPage],
    tmp_path: Path,
    raises: Exception,
    code: int,
    line: str,
) -> None:
    def serve(page: PlatePage, *, port: int) -> None:
        raise raises

    _stand_in(monkeypatch, serve)
    result = _plate(doc=tmp_path / "run.json")
    assert result.exit_code == code, result.output
    assert line in result.output.splitlines()
    assert spy_page.closed == ["close"]
    assert closes == fake_stand


def test_a_run_that_is_going_when_the_server_ends_is_stopped_and_said_last(
    monkeypatch: pytest.MonkeyPatch,
    fake_stand: list[Microscope],
    closes: list[Microscope],
    spy_page: type[_SpyPage],
    tmp_path: Path,
) -> None:
    def serve(page: PlatePage, *, port: int) -> None:
        page.run("slow", [])
        assert Slow.started.wait(5)

    _stand_in(monkeypatch, serve)
    result = _plate(doc=tmp_path / "run.json")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "slow was stopped"
    assert spy_page.order == ["stop"]
    assert closes == fake_stand


def test_a_second_ctrl_c_during_close_still_releases_the_stand(
    monkeypatch: pytest.MonkeyPatch,
    fake_stand: list[Microscope],
    closes: list[Microscope],
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Interrupted(_SpyPage):
        def stop(self) -> None:
            events.append("stop")
            super().stop()

        def join(self, timeout_s: float | None = None) -> bool:
            events.append("join")
            raise KeyboardInterrupt  # the operator's second Ctrl-C, from the stub (FM-44)

    monkeypatch.setattr(controller_module, "PlatePage", Interrupted)

    def serve(page: PlatePage, *, port: int) -> None:
        page.run("slow", [])
        assert Slow.started.wait(5)

    _stand_in(monkeypatch, serve)
    result = _plate(doc=tmp_path / "run.json")
    # The stop was sent before the wait, and the stand was released after it.
    assert events == ["stop", "join"]
    assert closes == fake_stand
    assert result.exit_code == 130, result.output
    assert "✗ interrupted" in result.output
    assert "slow was stopped" not in result.output


def test_smc_starts_without_importing_panel() -> None:
    code = (
        "import sys, smc.cli\n"
        "loaded = [m for m in ('panel', 'bokeh', 'smc.ui.plate_page', 'smc.ui.controller') "
        "if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_smc_ui_lists_its_pages() -> None:
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0, result.output
    assert re.search(r"\bplate\b", _plain(result))
