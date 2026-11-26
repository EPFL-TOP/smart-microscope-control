"""The model server without HTTP: what it lists, what it answers, how it loads its models, and what it never imports (inference.md §8.6).

``ModelServer`` is exercised on stub models, and ``prepare`` with an injected loader or
with the real one on a folder with no weights: no torch, no kit, no weights (FM-40). The
real server behind HTTP is ``tests/test_model_server_kit.py``'s and runs only where the
kit is installed.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import smc
from smc.inference import models as models_mod
from smc.inference import served as served_mod
from smc.inference import server as server_mod
from smc.inference.cards import ModelCard, model_card
from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, model_identity
from smc.inference.preprocessing import ModelPreprocessing
from smc.inference.served import decode_answer, decode_listing
from smc.inference.server import SERVER_SCORE_FLOOR, ModelServer, prepare
from smc.testing import StubDetector, stub_model

SRC = Path(smc.__file__).resolve().parent
NATIVE = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
SIZED = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=0.5)
SHA_A, SHA_B = "a" * 64, "b" * 64
#: What the kit and its server stack are, by the names the source imports them under.
HEAVY = ("imaging_server_kit", "uvicorn", "fastapi")


def _models(**kwargs: Any) -> dict[str, ResolvedModel]:
    """Two stub models: ``alpha`` on cuda with three classes, ``beta`` on cpu with one."""
    alpha = StubDetector(
        {7: [Detection(1.0, 2.0, 3.0, 4.0, "single", float(np.float32(0.93)))]},
        device="cuda",
        **kwargs,
    )
    beta = StubDetector(classes=("blob",), device="cpu")
    return {
        "alpha": stub_model(
            alpha, name="alpha", weights_sha256=SHA_A, preprocessing=NATIVE
        ),
        "beta": stub_model(
            beta, name="beta", weights_sha256=SHA_B, preprocessing=NATIVE
        ),
    }


def _frame(key: int = 7, shape: tuple[int, int] = (30, 40)) -> npt.NDArray[Any]:
    return np.full(shape, key, dtype=np.uint16)


def test_the_listing_names_each_model_with_its_identity_classes_and_device() -> None:
    listed = decode_listing(ModelServer(_models()).listing())

    assert [(m.name, m.identity, m.classes, m.device) for m in listed] == [
        (
            "alpha",
            model_identity(SHA_A, NATIVE),
            ("single", "doublet", "debris"),
            "cuda",
        ),
        ("beta", model_identity(SHA_B, NATIVE), ("blob",), "cpu"),
    ]


def test_an_answer_carries_the_models_identity_and_the_detections_exactly() -> None:
    answer = decode_answer(ModelServer(_models()).detect("alpha", _frame(7), 0.275))

    assert answer.model == "alpha"
    assert answer.identity == model_identity(SHA_A, NATIVE)
    assert answer.device == "cuda"
    # A float32 score is a float64 that JSON writes at its full repr.
    assert answer.detections == (
        Detection(1.0, 2.0, 3.0, 4.0, "single", float(np.float32(0.93))),
    )
    assert answer.detections[0].score != 0.93


def test_a_frame_reaches_the_detector_as_it_was_sent() -> None:
    seen: list[npt.NDArray[Any]] = []
    stub = StubDetector(on_frame=lambda key: None)
    original = stub.detect

    def spy(frame: npt.NDArray[Any]) -> list[Detection]:
        seen.append(frame)
        return original(frame)

    stub.detect = spy  # type: ignore[method-assign]
    server = ModelServer({"m": stub_model(stub, name="m")})

    server.detect("m", _frame(3, (5, 9)).astype(np.float32), 0.0)

    assert seen[0].dtype == np.float32
    assert seen[0].shape == (5, 9)


def test_a_call_for_a_model_it_does_not_serve_names_what_it_serves() -> None:
    server = ModelServer(_models())

    with pytest.raises(ValueError, match="does not serve 'gamma'") as caught:
        server.detect("gamma", _frame(), 0.0)
    assert "it serves: alpha, beta" in str(caught.value)

    with pytest.raises(ValueError, match="model is required"):
        server.detect("", _frame(), 0.0)


def test_a_call_to_a_server_with_no_model_says_so() -> None:
    with pytest.raises(ValueError, match="it serves: no model"):
        ModelServer({}).detect("alpha", _frame(), 0.0)


@pytest.mark.parametrize(
    "shape", [(4, 5, 3), (4,), ()], ids=["colour", "one-dimensional", "scalar"]
)
def test_a_frame_that_is_not_two_dimensional_is_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="2-D"):
        ModelServer(_models()).detect("alpha", np.zeros(shape, np.uint16), 0.0)


@pytest.mark.parametrize("size", [-0.1, float("nan"), float("inf"), True])
def test_a_negative_or_missing_pixel_size_is_refused(size: float) -> None:
    with pytest.raises(
        ValueError, match="pixel_size_um must be a number of at least 0"
    ):
        ModelServer(_models()).detect("alpha", _frame(), size)


def test_a_pixel_size_of_zero_is_unknown_and_served() -> None:
    assert decode_answer(ModelServer(_models()).detect("alpha", _frame(), 0.0))


def test_no_refusal_holds_a_path_separator() -> None:
    server = ModelServer(_models())
    refusals: list[str] = []
    for args in (
        ("gamma", _frame(), 0.0),
        ("", _frame(), 0.0),
        ("alpha", np.zeros((2, 2, 2)), 0.0),
        ("alpha", _frame(), -1.0),
    ):
        with pytest.raises(
            ValueError, match=r"2-D|model is required|does not serve|pixel_size_um"
        ) as caught:
            server.detect(*args)
        refusals.append(str(caught.value))

    assert refusals
    assert not [text for text in refusals if "/" in text or "\\" in text]


class _SpyLock:
    """A lock that says when someone arrives while it is held: the wait is seen, not timed (FM-44)."""

    def __init__(self, contended: threading.Event) -> None:
        self._lock = threading.Lock()
        self._contended = contended

    def __enter__(self) -> None:
        if self._lock.locked():
            self._contended.set()
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


class _HoldingDetector:
    """Holds the call for frame 0 until the test lets it go; records who is inside when."""

    classes = ("single",)
    device = "cpu"

    def __init__(self, entered: threading.Event, hold: threading.Event) -> None:
        self.events: list[tuple[str, int]] = []
        self._entered = entered
        self._hold = hold

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        key = int(frame[0, 0])
        self.events.append(("enter", key))
        if key == 0:
            self._entered.set()
            assert self._hold.wait(10), "the first call was never released"
        self.events.append(("leave", key))
        return []


def test_two_calls_are_served_one_at_a_time() -> None:
    entered, hold, contended = threading.Event(), threading.Event(), threading.Event()
    detector = _HoldingDetector(entered, hold)
    server = ModelServer({"m": stub_model(detector, name="m")})
    server._lock = _SpyLock(contended)  # type: ignore[assignment]
    first = threading.Thread(target=server.detect, args=("m", _frame(0), 0.0))
    second = threading.Thread(target=server.detect, args=("m", _frame(1), 0.0))
    first.start()
    try:
        assert entered.wait(10), "the first call never reached the detector"
        second.start()
        # The second call has reached the lock while the first is inside: only
        # now is the first let go.
        assert contended.wait(10), "the second call did not wait for the first"
    finally:
        hold.set()
        first.join(10)
        second.join(10)
    assert not first.is_alive()
    assert not second.is_alive()
    assert detector.events == [
        ("enter", 0),
        ("leave", 0),
        ("enter", 1),
        ("leave", 1),
    ]


# --- prepare ----------------------------------------------------------------------


def _card(name: str, rule: ModelPreprocessing = NATIVE, sha: str = SHA_A) -> ModelCard:
    return ModelCard(
        name=name,
        weights=f"{name}.pth",
        weights_sha256=sha,
        classes=("single", "doublet", "debris"),
        trained_um_per_px=None,
        preprocessing=rule,
    )


def _cards(monkeypatch: pytest.MonkeyPatch, *cards: ModelCard) -> None:
    """Make the package's cards these: ``prepare`` reads them by name."""
    by_name = {card.name: card for card in cards}

    def card_of(name: str) -> ModelCard:
        if name not in by_name:
            raise ModelError(
                f"no model named {name!r}; the models are: {sorted(by_name)}"
            )
        return by_name[name]

    monkeypatch.setattr(server_mod, "model_names", lambda: tuple(sorted(by_name)))
    monkeypatch.setattr(server_mod, "model_card", card_of)


class _Loader:
    """A stand-in for ``load_local_model`` that records its calls; a name in ``refuse`` fails."""

    def __init__(self, refuse: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[ModelCard, dict[str, Any]]] = []
        self._refuse = refuse or {}

    def __call__(self, card: ModelCard, **kwargs: Any) -> ResolvedModel:
        self.calls.append((card, kwargs))
        if card.name in self._refuse:
            raise ModelError(self._refuse[card.name])
        return stub_model(
            StubDetector(classes=card.classes, device="cuda"),
            name=card.name,
            weights_sha256=card.weights_sha256,
            preprocessing=kwargs["preprocessing"],
        )


def test_prepare_loads_every_card_with_its_own_rule_and_the_floor_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = ModelPreprocessing(percentiles=(1.0, 99.0), model_um_per_px=None)
    _cards(monkeypatch, _card("alpha"), _card("beta", other, SHA_B))
    loader = _Loader()

    server, notes = prepare(None, device="cpu", load=loader)

    assert notes == []
    assert [card.name for card, _ in loader.calls] == ["alpha", "beta"]
    for card, kwargs in loader.calls:
        assert kwargs["preprocessing"] == card.preprocessing
        assert kwargs["score_threshold"] == SERVER_SCORE_FLOOR == 0.0
        assert kwargs["device"] == "cpu"
        assert kwargs["weights"] is None
        # A native rule never reads these; they are the placeholders.
        assert kwargs["frame_shape_px"] == (1, 1)
        assert kwargs["frame_um_per_px"] == 1.0
    listed = decode_listing(server.listing())
    assert [(m.name, m.identity) for m in listed] == [
        ("alpha", model_identity(SHA_A, NATIVE)),
        ("beta", model_identity(SHA_B, other)),
    ]


def test_prepare_serves_only_the_cards_it_is_given_once_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cards(monkeypatch, _card("alpha"), _card("beta", sha=SHA_B))
    loader = _Loader()

    server, notes = prepare(["beta", "beta"], device="auto", load=loader)

    assert [card.name for card, _ in loader.calls] == ["beta"]
    assert [m.name for m in decode_listing(server.listing())] == ["beta"]
    assert notes == []


def test_a_card_that_cannot_be_served_is_skipped_with_a_note_when_no_name_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cards(
        monkeypatch,
        _card("alpha"),
        _card("noweights", sha=SHA_B),
        _card("sized", SIZED, "c" * 64),
    )
    loader = _Loader(refuse={"noweights": "no model at /srv/models/noweights.pth"})

    server, notes = prepare(None, device="cpu", load=loader)

    assert [m.name for m in decode_listing(server.listing())] == ["alpha"]
    assert len(notes) == 2
    by_card = {note.split(" ", 1)[0]: note for note in notes}
    # The notes go to the server's own console: they may name a path, and do here.
    assert "/srv/models/noweights.pth" in by_card["noweights"]
    assert "0.5 um per pixel" in by_card["sized"]
    # The sized card was refused before it was loaded.
    assert "sized" not in [card.name for card, _ in loader.calls]


def test_a_named_card_that_cannot_be_served_is_a_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cards(
        monkeypatch,
        _card("alpha"),
        _card("noweights", sha=SHA_B),
        _card("sized", SIZED, "c" * 64),
    )
    loader = _Loader(refuse={"noweights": "no model at the-place"})

    with pytest.raises(ModelError, match="no model at the-place"):
        prepare(["alpha", "noweights"], device="cpu", load=loader)
    with pytest.raises(ModelError, match=r"0\.5 um per pixel"):
        prepare(["sized"], device="cpu", load=loader)
    with pytest.raises(ModelError, match="no model named 'nope'"):
        prepare(["nope"], device="cpu", load=loader)


def test_no_model_at_all_is_a_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, _card("alpha"))
    loader = _Loader(refuse={"alpha": "no model at the-place"})

    with pytest.raises(ModelError, match="no model can be served here") as caught:
        prepare(None, device="cpu", load=loader)
    assert "alpha is not served: no model at the-place" in str(caught.value)

    _cards(monkeypatch)
    with pytest.raises(ModelError, match="there is no model card"):
        prepare(None, device="cpu", load=loader)


def test_prepare_never_goes_through_load_model_even_with_a_server_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A server started in an environment that has SMC_MODEL_SERVER (the PC's own, or
    # a copy of the acquisition PC's) must not become a client of itself: `prepare`
    # loads with `load_local_model`, so no wire is ever opened.
    opened: list[str] = []

    def open_wire(address: str) -> object:
        opened.append(address)
        raise AssertionError("the server went looking for a model server")

    monkeypatch.setattr(served_mod, "open_wire", open_wire)
    monkeypatch.setenv(served_mod.SERVER_ENV, "http://127.0.0.1:9")
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))

    with pytest.raises(ModelError) as caught:
        prepare(["cell-classes-v2"], device="cpu")

    # The local loader's own words: it looked for the weights on this PC.
    assert "no model at" in str(caught.value)
    assert opened == []


def test_the_default_loader_runs_the_model_here_and_names_the_missing_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `load_local_model` is prepare's default: on a folder with no weights the
    # real card of the repository is refused with the loader's own words.
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))
    card = model_card("cell-classes-v2")

    with pytest.raises(ModelError) as caught:
        prepare(["cell-classes-v2"], device="cpu")

    assert f"no model at {tmp_path / card.weights}" in str(caught.value)
    assert models_mod.load_local_model is not models_mod.load_model


# --- what the server never imports ------------------------------------------------


def _heavy_imports(source: str) -> list[tuple[int, bool]]:
    """The imports of the kit, uvicorn and fastapi in ``source``: (line, inside a function)."""
    found: list[tuple[int, bool]] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        inside = in_function or isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        )
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] if node.level == 0 else []
        else:
            names = []
        if any(name.split(".")[0] in HEAVY for name in names):
            found.append((node.lineno, in_function))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(ast.parse(source), False)
    return found


def test_the_walk_sees_a_module_level_and_a_nested_import() -> None:
    # The walk is the only thing that fails the test below if it is blind.
    source = (
        "import uvicorn\n"
        "def f():\n"
        "    from imaging_server_kit import algorithm_server\n"
        "    import fastapi.responses\n"
        "class C:\n"
        "    import fastapi\n"
        "import numpy\n"
        "from smc.inference import server\n"
    )

    assert _heavy_imports(source) == [(1, False), (3, True), (4, True), (6, False)]


def test_the_kit_is_imported_only_in_server_py_inside_a_function() -> None:
    walked = 0
    outside: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        walked += 1
        for line, in_function in _heavy_imports(path.read_text(encoding="utf-8")):
            if not (path == SRC / "inference" / "server.py" and in_function):
                outside.append(f"{path.relative_to(SRC)}:{line}")
    assert walked > 50
    assert outside == []


def test_served_and_server_import_without_the_kit_torch_uvicorn_and_fastapi(
    tmp_path: Path,
) -> None:
    blocked = ("torch", "torchvision", *HEAVY)
    code = (
        "import json, sys\n"
        f"for name in {blocked!r}:\n"
        "    sys.modules[name] = None\n"
        "import smc.inference.served\n"
        "import smc.inference.server\n"
        "print(json.dumps(sorted(m for m in sys.modules\n"
        f"    if m.split('.')[0] in {blocked!r} and sys.modules[m] is not None)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
