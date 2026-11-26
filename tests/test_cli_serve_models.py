"""``smc serve-models`` (inference.md §8.6): the order of its checks, what it prints, and how it ends.

``prepare`` and ``serve`` of ``smc.inference.server`` are replaced by stand-ins with a
record of their own, so none of this needs the kit, uvicorn, torch or weights, and it
runs on every CI job; the real server is ``tests/test_model_server_kit.py``'s. The
runner pins the width (FM-37) and clears the variables a developer's PC may have set
(FM-40). A taken port is held by the test itself, on the address the command probes
(FM-144, FM-150); an interrupt is raised from the stand-in, never sent as a signal
(FM-44).
"""

from __future__ import annotations

import errno
import socket
import sys
from collections.abc import Sequence
from typing import Any

import pytest
from typer.testing import CliRunner, Result

import smc.cli as cli
from smc.cli import app
from smc.inference import server as server_mod
from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, model_identity
from smc.inference.preprocessing import ModelPreprocessing
from smc.inference.server import ModelServer
from smc.testing import StubDetector, stub_model

runner = CliRunner(
    env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_MODEL_SERVER": None}
)

FIX = 'install the serve extra: pip install -e ".[serve]"'
NATIVE = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
SHA = "a" * 64


class _Stand:
    """What the stand-ins saw: the order of the calls, and what each was given."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prepared: list[tuple[Sequence[str] | None, str]] = []
        self.served: list[tuple[ModelServer, str, int]] = []
        self.prepare_error: Exception | None = None
        self.serve_error: BaseException | None = None
        self.notes: list[str] = []


def _models() -> dict[str, ResolvedModel]:
    stub = StubDetector(
        {1: [Detection(1.0, 2.0, 3.0, 4.0, "single", 0.9)]}, device="cuda"
    )
    return {
        "alpha": stub_model(
            stub, name="alpha", weights_sha256=SHA, preprocessing=NATIVE
        )
    }


@pytest.fixture
def stand(monkeypatch: pytest.MonkeyPatch) -> _Stand:
    """Replace the extra's check, ``prepare`` and ``serve`` of the server module."""
    seen = _Stand()

    def check_serve_extra() -> None:
        seen.calls.append("check")

    def prepare(
        names: Sequence[str] | None, *, device: str
    ) -> tuple[ModelServer, list[str]]:
        seen.calls.append("prepare")
        seen.prepared.append((names, device))
        if seen.prepare_error is not None:
            raise seen.prepare_error
        return ModelServer(_models()), list(seen.notes)

    def serve(server: ModelServer, *, host: str, port: int) -> None:
        seen.calls.append("serve")
        seen.served.append((server, host, port))
        if seen.serve_error is not None:
            raise seen.serve_error

    monkeypatch.setattr(server_mod, "check_serve_extra", check_serve_extra)
    monkeypatch.setattr(server_mod, "prepare", prepare)
    monkeypatch.setattr(server_mod, "serve", serve)
    return seen


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _serve(*args: str) -> Result:
    return runner.invoke(app, ["serve-models", *args])


def test_a_held_port_is_refused_before_any_model_is_loaded(stand: _Stand) -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        result = _serve("--port", str(port))

    assert result.exit_code == 2, result.output
    assert (
        f"✗ port {port} is in use; choose another with --port"
        in result.output.splitlines()
    )
    # A model is 166 MB and seconds of torch: the port is looked at first (FM-144).
    assert "prepare" not in stand.calls
    assert "serve" not in stand.calls


def test_a_port_held_on_ipv6_only_is_refused_for_localhost(stand: _Stand) -> None:
    # `localhost` is two addresses on most hosts (FM-150): a probe of one passes it.
    try:
        held = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        held.bind(("::1", 0))
    except OSError:
        pytest.skip("this host has no IPv6 loopback")
    with held:
        port = held.getsockname()[1]
        result = _serve("--host", "localhost", "--port", str(port))

    assert result.exit_code == 2, result.output
    assert "prepare" not in stand.calls


def test_an_address_this_host_cannot_bind_is_not_a_taken_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _free_port()
    real = socket.socket

    def make(
        family: int = -1, kind: int = -1, proto: int = -1, fileno: Any = None
    ) -> Any:
        if family == socket.AF_INET6:
            raise OSError(errno.EAFNOSUPPORT, "no IPv6 here")
        return real(family, kind, proto, fileno)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ],
    )
    monkeypatch.setattr(socket, "socket", make)

    assert not cli._port_is_taken_on("localhost", port)


def test_a_port_is_taken_on_the_address_that_is_asked_about() -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        assert cli._port_is_taken_on("127.0.0.1", port)
    assert not cli._port_is_taken_on("127.0.0.1", _free_port())


def test_a_missing_kit_says_to_install_the_serve_extra(
    monkeypatch: pytest.MonkeyPatch, stand: _Stand
) -> None:
    # The stand-in of the check is put back to the real one, and the kit is made
    # missing as it is on a PC without the extra.
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "imaging_server_kit", None)
    prepared: list[object] = []
    monkeypatch.setattr(server_mod, "prepare", lambda *a, **k: prepared.append(a))

    result = _serve("--port", str(_free_port()))

    assert result.exit_code == 1, result.output
    assert f"✗ {FIX}" in result.output.splitlines()
    assert prepared == []


def test_it_lists_the_models_it_serves_with_their_identities(stand: _Stand) -> None:
    stand.notes = ["beta is not served: no model at somewhere"]
    port = _free_port()

    result = _serve("--port", str(port))

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "beta is not served: no model at somewhere" in lines
    identity = model_identity(SHA, NATIVE)
    served_line = [line for line in lines if line.startswith("serving alpha")]
    assert served_line == [f"serving alpha  on cuda  identity {identity}"]
    # The notes come first, then the models, then the server starts.
    assert lines.index("beta is not served: no model at somewhere") < lines.index(
        served_line[0]
    )
    assert stand.calls == ["check", "prepare", "serve"]
    assert stand.prepared == [(None, "auto")]
    ((_, host, got_port),) = stand.served
    assert (host, got_port) == ("127.0.0.1", port)


def test_the_models_and_the_device_asked_for_reach_prepare(stand: _Stand) -> None:
    # The names are checked against the repository's cards before anything is loaded,
    # so the one card there is is the name to give.
    result = _serve(
        "--model",
        "cell-classes-v2",
        "--device",
        "cpu",
        "--port",
        str(_free_port()),
    )

    assert result.exit_code == 0, result.output
    assert stand.prepared == [(["cell-classes-v2"], "cpu")]


@pytest.mark.parametrize(
    ("host", "warns"),
    [
        ("0.0.0.0", True),
        ("192.0.2.1", True),
        ("lab-gpu.example", True),
        ("127.0.0.1", False),
        ("localhost", False),
        ("::1", False),
    ],
)
def test_a_host_that_is_not_loopback_warns(
    monkeypatch: pytest.MonkeyPatch, stand: _Stand, host: str, warns: bool
) -> None:
    # The resolver and the probe are not the test's subject here: an address that
    # cannot be bound, or a name that does not resolve, must not decide whether the
    # warning is printed.
    monkeypatch.setattr(cli, "_port_is_taken_on", lambda host, port: False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])

    result = _serve("--host", host, "--port", "8000")

    assert result.exit_code == 0, result.output
    said = (
        f"listening on {host}: anyone who reaches this port can run the models; "
        "it has no login"
    )
    assert (said in result.output.splitlines()) is warns


def test_a_host_that_does_not_resolve_is_refused_before_any_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch, stand: _Stand
) -> None:
    def no_such_host(*args: object, **kwargs: object) -> list[object]:
        raise socket.gaierror(socket.EAI_NONAME, "no such host")

    monkeypatch.setattr(socket, "getaddrinfo", no_such_host)

    result = _serve("--host", "gpu.nowhere.invalid", "--port", "8000")

    assert result.exit_code == 2, result.output
    assert (
        "✗ gpu.nowhere.invalid is not an address this PC knows; choose another "
        "with --host" in result.output.splitlines()
    )
    assert "prepare" not in stand.calls


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_host_is_refused_and_never_means_every_interface(
    stand: _Stand, blank: str
) -> None:
    result = _serve("--host", blank, "--port", "8000")

    assert result.exit_code == 2, result.output
    assert "--host is empty" in result.output
    assert stand.calls == []


def test_a_named_model_that_cannot_be_loaded_ends_with_its_diagnosis(
    stand: _Stand,
) -> None:
    stand.prepare_error = ModelError("no model at /srv/alpha.pth; put alpha.pth there")

    result = _serve("--model", "cell-classes-v2", "--port", str(_free_port()))

    assert result.exit_code == 1, result.output
    assert (
        "✗ no model at /srv/alpha.pth; put alpha.pth there"
        in result.output.splitlines()
    )
    assert "serve" not in stand.calls


def test_an_unknown_model_name_lists_the_known_ones(stand: _Stand) -> None:
    result = _serve("--model", "nope", "--port", str(_free_port()))

    assert result.exit_code == 2, result.output
    assert "no model named 'nope'; the models are: cell-classes-v2" in result.output
    # Refused from the cards alone: no model was loaded for it.
    assert "prepare" not in stand.calls


def test_an_unknown_device_is_refused_with_the_known_ones(stand: _Stand) -> None:
    result = _serve("--device", "tpu", "--port", str(_free_port()))

    assert result.exit_code == 2, result.output
    assert "auto, cpu, cuda or mps" in result.output
    assert stand.calls == []


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_a_port_out_of_range_is_refused(stand: _Stand, port: str) -> None:
    result = _serve("--port", port)

    assert result.exit_code == 2, result.output
    assert stand.calls == []


def test_a_server_that_ends_when_interrupted_again_exits_130(stand: _Stand) -> None:
    # uvicorn handles the first Ctrl-C itself and `serve` returns (exit 0, above);
    # an interrupt that reaches the command is the operator's second one (FM-143).
    stand.serve_error = KeyboardInterrupt()

    result = _serve("--port", str(_free_port()))

    assert result.exit_code == 130, result.output


def test_smc_starts_without_importing_the_server_module() -> None:
    import subprocess

    code = "import sys\nimport smc.cli\nprint('smc.inference.server' in sys.modules)\n"
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "False"
