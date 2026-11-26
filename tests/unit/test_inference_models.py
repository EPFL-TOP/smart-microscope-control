"""A model's identity and how a card becomes a detector: where the weights are, what is refused, what each failure says (inference.md §4, §5).

None of it needs torch or the real weights, so CI runs it: the weights are a few
bytes whose SHA-256 the test's own card names, and ``TorchDetector`` is a
stand-in that records how it was built. The test of the card against the real
file is skipped without it and run by hand with ``SMC_MODELS`` and a venv that
has torch (docs/design/inference.md §1, §7).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import smc
from smc.inference import local as local_mod
from smc.inference import models as models_mod
from smc.inference.blobs import BLOB_DETECTOR, BLOB_DETECTOR_IDENTITY, blob_model
from smc.inference.cards import DEFAULT_MODEL_NAME, ModelCard, model_card
from smc.inference.detector import Detector
from smc.inference.errors import ModelError
from smc.inference.models import (
    ResolvedModel,
    file_sha256,
    load_model,
    model_identity,
    weights_folder,
)
from smc.inference.preprocessing import (
    PREPROCESSING_VERSION,
    ModelPreprocessing,
    preprocessing_for,
    preprocessing_sha256,
)

CHECKOUT = Path(smc.__file__).resolve().parents[2]
CLASSES = ("single", "doublet", "debris")
NATIVE = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
#: The three digests of inference.md §4's table, computed 2026-11-26 from the
#: canonical text; a copy, so that a change of the text or of the version fails.
NATIVE_DIGEST = "2df8feb4070d6d7b527502090c9f1be7003a1b50b7ee09390b81b8413a6cfff1"
INTS_DIGEST = "6b31dfc76ff951636112c94fe6e220013f8e9f547d98929e93304b00e0e16417"
NONE_DIGEST = "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"


# --- helpers ------------------------------------------------------------------


def _card(
    sha: str, *, classes: tuple[str, ...] = CLASSES, weights: str = "w.pth"
) -> ModelCard:
    return ModelCard(
        name="m",
        weights=weights,
        weights_sha256=sha,
        classes=classes,
        trained_um_per_px=None,
        preprocessing=NATIVE,
    )


def _weights(tmp_path: Path, content: bytes = b"not a checkpoint") -> tuple[Path, str]:
    """A file standing for the weights, and its SHA-256."""
    path = tmp_path / "w.pth"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _load(
    card: ModelCard,
    *,
    weights: Path | None = None,
    rule: ModelPreprocessing = NATIVE,
    device: str = "cpu",
    shape: tuple[int, int] = (1412, 1412),
    um: float = 0.2756,
) -> ResolvedModel:
    return load_model(
        card,
        preprocessing=rule,
        frame_shape_px=shape,
        frame_um_per_px=um,
        device=device,
        score_threshold=0.5,
        weights=weights,
    )


def _reason(card: ModelCard, **kwargs: Any) -> str:
    with pytest.raises(ModelError) as caught:
        _load(card, **kwargs)
    return str(caught.value)


def _standin(
    monkeypatch: pytest.MonkeyPatch,
    *,
    classes: tuple[str, ...] = CLASSES,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Replace ``TorchDetector`` by a recorder; the list holds one dict per construction.

    The stand-in declares the classes and the device it is given, as the real one
    reads them from its checkpoint and resolves them, and finds nothing.
    """
    built: list[dict[str, Any]] = []

    class Standin:
        def __init__(self, weights: Path, **kwargs: Any) -> None:
            built.append({"weights": weights, **kwargs})
            self.classes = classes
            self.device = device

        def detect(self, frame: Any) -> list[Any]:
            return []

    monkeypatch.setattr(local_mod, "TorchDetector", Standin)
    return built


def _raising(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    class Raising:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise error

    monkeypatch.setattr(local_mod, "TorchDetector", Raising)


# --- the identity -------------------------------------------------------------


def test_the_preprocessing_digests_are_pinned() -> None:
    ints = ModelPreprocessing(percentiles=(1, 99), model_um_per_px=1)  # type: ignore[arg-type]

    assert PREPROCESSING_VERSION == 1
    assert preprocessing_sha256(NATIVE) == NATIVE_DIGEST
    assert preprocessing_sha256(ints) == INTS_DIGEST
    assert preprocessing_sha256(None) == NONE_DIGEST
    # The canonical text itself, so that a digest that moved says what changed.
    text = '{"model_um_per_px":null,"percentiles":[0.5,99.5],"version":1}'
    assert hashlib.sha256(text.encode("ascii")).hexdigest() == NATIVE_DIGEST
    assert hashlib.sha256(b"null").hexdigest() == NONE_DIGEST
    assert json.loads(text)["version"] == PREPROCESSING_VERSION


def test_the_identity_of_cell_classes_v2_is_its_weights_and_its_preprocessing() -> None:
    card = model_card(DEFAULT_MODEL_NAME)

    identity = model_identity(card.weights_sha256, card.preprocessing)

    assert identity == f"{card.weights_sha256}+{NATIVE_DIGEST}"
    assert len(identity) == 129


@pytest.mark.parametrize(
    "other",
    [
        ModelPreprocessing(percentiles=(1.0, 99.5), model_um_per_px=None),
        ModelPreprocessing(percentiles=(0.5, 99.0), model_um_per_px=None),
        ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=0.49),
    ],
    ids=["low", "high", "pixel size"],
)
def test_a_change_of_a_percentile_or_of_the_pixel_size_is_another_identity(
    other: ModelPreprocessing,
) -> None:
    sha = "a" * 64

    assert model_identity(sha, other) != model_identity(sha, NATIVE)
    assert model_identity(sha, other).startswith(sha + "+")
    # And other weights under the same rule are another identity too.
    assert model_identity("b" * 64, NATIVE) != model_identity(sha, NATIVE)


def test_an_int_and_a_float_in_a_rule_are_one_identity() -> None:
    as_ints = ModelPreprocessing(percentiles=(1, 99), model_um_per_px=1)  # type: ignore[arg-type]
    as_floats = ModelPreprocessing(percentiles=(1.0, 99.0), model_um_per_px=1.0)

    assert preprocessing_sha256(as_ints) == preprocessing_sha256(as_floats)


# --- the input size, worked out for a scan -------------------------------------


def test_a_native_rule_keeps_the_models_own_resize() -> None:
    shown = preprocessing_for(NATIVE, frame_shape_px=(60, 80), frame_um_per_px=0.5)

    assert shown.input_size_px is None
    assert shown.percentiles == (0.5, 99.5)


def test_a_rule_that_names_a_scale_gives_the_input_size_of_the_scan() -> None:
    """1412 px at 0.2756 um shown at 0.49 um is (794, 795): the figures of m3 §9.4."""
    rule = ModelPreprocessing(percentiles=(1.0, 99.0), model_um_per_px=0.49)

    shown = preprocessing_for(rule, frame_shape_px=(1412, 1412), frame_um_per_px=0.2756)

    assert shown.input_size_px == (794, 795)
    assert shown.percentiles == (1.0, 99.0)


# --- where the weights are ----------------------------------------------------


def test_smc_models_sets_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))

    assert weights_folder() == tmp_path


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_or_blank_smc_models_is_not_the_working_directory(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Path("")`` is ``.``: an empty variable must not make the cwd the models folder (FM-78)."""
    monkeypatch.setenv("SMC_MODELS", value)

    assert weights_folder() == CHECKOUT / "local" / "models"


def test_without_smc_models_the_weights_are_in_the_checkouts_local_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SMC_MODELS", raising=False)

    assert weights_folder() == CHECKOUT / "local" / "models"


def test_the_weights_are_looked_for_in_the_models_folder_by_the_cards_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, sha = _weights(tmp_path)
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))
    built = _standin(monkeypatch)

    resolved = _load(_card(sha))

    assert built[0]["weights"] == tmp_path / "w.pth"
    assert resolved.weights_sha256 == sha


def test_a_model_path_replaces_the_cards_file_but_not_its_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = elsewhere / "renamed.pth"
    path.write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()
    monkeypatch.setenv("SMC_MODELS", str(tmp_path / "nothing-here"))
    built = _standin(monkeypatch)

    _load(_card(sha), weights=path)
    assert built[0]["weights"] == path

    reason = _reason(_card("0" * 64), weights=path)
    assert sha in reason
    assert "renamed.pth" in reason


def test_a_missing_weights_file_says_where_it_looked_and_names_the_cards_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))
    built = _standin(monkeypatch)

    reason = _reason(_card("0" * 64, weights="cells-v9.pth"))

    assert str(tmp_path / "cells-v9.pth") in reason
    assert "cells-v9.pth" in reason
    assert "SMC_MODELS" in reason
    assert "model_path" in reason
    assert built == []


def test_a_missing_model_path_is_told_the_same_way(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pth"

    reason = _reason(_card("0" * 64), weights=missing)

    assert str(missing) in reason
    assert "SMC_MODELS" in reason


def test_a_folder_the_os_refuses_to_stat_is_a_model_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder the user cannot enter, a share that dropped: ``is_file`` raises, it does not say no (FM-116)."""
    weights = tmp_path / "w.pth"
    real = Path.is_file

    def refused(self: Path) -> bool:
        if self == weights:
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "is_file", refused)

    reason = _reason(_card("0" * 64), weights=weights)

    assert "Permission denied" in reason
    assert str(weights) in reason
    assert "SMC_MODELS" in reason
    assert "w.pth" in reason


def test_a_weights_file_that_cannot_be_read_says_who_may_hold_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = _weights(tmp_path)

    def locked(_: Path) -> str:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(models_mod, "file_sha256", locked)

    reason = _reason(_card(sha), weights=path)

    assert reason.startswith("w.pth cannot be read to hash it")
    assert "Permission denied" in reason
    assert "another program" in reason


# --- the weights against the card ---------------------------------------------


def test_weights_of_another_sha256_are_refused_with_both_values_before_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = _weights(tmp_path)
    wanted = "f" * 64
    built = _standin(monkeypatch)

    reason = _reason(_card(wanted), weights=path)

    assert sha in reason
    assert wanted in reason
    assert "w.pth" in reason
    assert "not the model the card describes" in reason
    assert built == []  # the model is never built, and torch never imported


def test_the_weights_are_hashed_in_pieces(tmp_path: Path) -> None:
    content = bytes(range(256)) * (3 * 1024 * 4 + 5)  # a little over 3 MiB
    path = tmp_path / "big.pth"
    path.write_bytes(content)
    sizes: list[int] = []
    real_open = Path.open

    class Spy:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> Spy:
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.close()

        def read(self, size: int = -1) -> bytes:
            sizes.append(size)
            data: bytes = self._handle.read(size)
            return data

    def spied(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return Spy(handle) if self == path else handle

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", spied)
        digest = file_sha256(path)

    assert len(content) > 3 * 1024 * 1024
    assert digest == hashlib.sha256(content).hexdigest()
    assert max(sizes) == 1 << 20
    assert len(sizes) >= 4


def test_a_refused_input_size_is_told_before_the_weights_are_looked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No file exists: a size the model is never given must not wait for 166 MB."""
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))
    rule = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=0.01)

    reason = _reason(_card("0" * 64), rule=rule)

    assert "model_um_per_px=0.01" in reason
    assert "no model at" not in reason


# --- what a failure of torch says ---------------------------------------------


def test_a_missing_torch_says_to_install_the_detect_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = _weights(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)

    reason = _reason(_card(sha), weights=path)

    assert "torch is not installed" in reason
    assert "pip install -e .[detect]" in reason


def test_a_torch_that_does_not_load_is_told_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows raises ``OSError`` for a DLL that does not load: reinstalling is the cure."""
    path, sha = _weights(tmp_path)
    _raising(
        monkeypatch, OSError("[WinError 126] The specified module could not be found")
    )

    reason = _reason(_card(sha), weights=path)

    assert "did not load" in reason
    assert "WinError 126" in reason
    assert "not installed" not in reason
    # torch is imported in the process that already holds the stand's DLLs (FM-95):
    # a fresh console tells that conflict from a broken install.
    assert "import torch" in reason


def test_a_refused_checkpoint_is_a_model_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = _weights(tmp_path)
    _raising(monkeypatch, ValueError("w.pth has no classes"))

    assert _reason(_card(sha), weights=path) == "w.pth has no classes"


def test_an_import_error_that_is_not_a_missing_module_is_not_told_as_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DLL load failed while importing _imaging`` is an ``ImportError`` too."""
    path, sha = _weights(tmp_path)
    _raising(monkeypatch, ImportError("DLL load failed while importing _imaging"))

    reason = _reason(_card(sha), weights=path)

    assert "DLL load failed while importing _imaging" in reason
    assert "could not be imported" in reason
    assert "not installed" not in reason
    assert "pip install -e .[detect]" in reason


def test_a_runtime_error_while_importing_torch_is_told_as_not_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torch and a torchvision of versions that do not go together raise ``RuntimeError``."""
    path, sha = _weights(tmp_path)
    _raising(monkeypatch, RuntimeError("operator torchvision::nms does not exist"))

    reason = _reason(_card(sha), weights=path)

    assert "did not load" in reason
    assert "torchvision::nms" in reason


def test_the_cause_is_chained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, sha = _weights(tmp_path)
    _raising(monkeypatch, ValueError("why"))

    with pytest.raises(ModelError) as caught:
        _load(_card(sha), weights=path)

    assert isinstance(caught.value.__cause__, ValueError)


def test_weights_whose_classes_differ_from_the_card_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = _weights(tmp_path)
    _standin(monkeypatch, classes=("single", "doublet", "dead"))

    reason = _reason(_card(sha), weights=path)

    assert "w.pth" in reason
    assert "['single', 'doublet', 'dead']" in reason
    assert "['single', 'doublet', 'debris']" in reason

    # The order is the checkpoint's, so a card that lists them otherwise is refused too.
    _standin(monkeypatch)
    assert "classes" in _reason(
        _card(sha, classes=("doublet", "single", "debris")), weights=path
    )


# --- the resolved model -------------------------------------------------------


def test_load_model_runs_locally_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The percentiles and the input size reach the detector: the seam of FM-124."""
    path, sha = _weights(tmp_path)
    rule = ModelPreprocessing(percentiles=(1.0, 99.0), model_um_per_px=0.49)
    built = _standin(monkeypatch, device="cuda")

    resolved = _load(_card(sha), weights=path, rule=rule, device="auto")

    assert resolved.name == "m"
    assert resolved.runs_on == "local:cuda"
    assert resolved.weights_sha256 == sha
    assert resolved.identity == model_identity(sha, rule)
    assert isinstance(resolved.detector, Detector)
    (call,) = built
    assert call["weights"] == path
    assert call["device"] == "auto"
    assert call["score_threshold"] == 0.5
    assert call["preprocessing"].percentiles == (1.0, 99.0)
    assert call["preprocessing"].input_size_px == (794, 795)


def test_the_identity_follows_the_rule_the_run_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = _weights(tmp_path)
    _standin(monkeypatch)
    other = ModelPreprocessing(percentiles=(1.0, 99.5), model_um_per_px=None)

    same = _load(_card(sha), weights=path)
    changed = _load(_card(sha), weights=path, rule=other)

    assert same.weights_sha256 == changed.weights_sha256
    assert same.identity != changed.identity


def test_a_resolved_model_is_frozen() -> None:
    resolved = blob_model()

    with pytest.raises(AttributeError):
        resolved.name = "other"  # type: ignore[misc]


def test_the_blob_model_has_its_own_identity() -> None:
    resolved = blob_model()

    digest = hashlib.sha256(BLOB_DETECTOR_IDENTITY.encode("utf-8")).hexdigest()
    assert resolved.name == BLOB_DETECTOR == "blobs"
    assert resolved.weights_sha256 == digest
    assert resolved.identity == f"{digest}+{NONE_DIGEST}"
    assert resolved.runs_on == "local:cpu"
    assert resolved.detector.classes == ("single", "doublet")


def test_the_blob_model_reads_its_identity_string_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A change of its rules is a new version in the string, and a resume across it is refused."""
    before = blob_model().identity
    monkeypatch.setattr(
        "smc.inference.blobs.BLOB_DETECTOR_IDENTITY", BLOB_DETECTOR_IDENTITY + " (2)"
    )

    assert blob_model().identity != before


# --- importing without torch --------------------------------------------------


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_every_module_of_smc_inference_imports_without_torch() -> None:
    """``smc run --list`` and the algorithm developer's kit work on a PC with no torch."""
    run = _run(
        """
        import importlib, pkgutil, sys
        sys.modules["torch"] = None
        sys.modules["torchvision"] = None
        import smc.inference
        names = sorted(
            m.name for m in pkgutil.walk_packages(smc.inference.__path__, "smc.inference.")
        )
        for name in names:
            importlib.import_module(name)
        print(",".join(names))
        """
    )

    assert run.returncode == 0, run.stderr
    modules = set(run.stdout.strip().split(","))
    assert {
        "smc.inference.blobs",
        "smc.inference.cards",
        "smc.inference.detector",
        "smc.inference.errors",
        "smc.inference.local",
        "smc.inference.models",
        "smc.inference.preprocessing",
    } <= modules


def test_importing_smc_inference_loads_neither_torch_nor_skimage() -> None:
    """The package's own import is light: ``BlobDetector`` and ``TorchDetector`` are imported from their modules."""
    run = _run(
        """
        import sys
        import smc.inference
        heavy = {"torch", "torchvision", "skimage"}
        print(sorted(k for k in sys.modules if k.split(".")[0] in heavy))
        """
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"


# --- the card against the real weights ----------------------------------------


def test_the_cell_classes_v2_card_matches_the_weights_on_this_pc() -> None:
    card = model_card(DEFAULT_MODEL_NAME)
    weights = weights_folder() / card.weights
    if not weights.is_file():
        pytest.skip(f"no weights at {weights} (set SMC_MODELS to their folder)")

    assert file_sha256(weights) == card.weights_sha256
