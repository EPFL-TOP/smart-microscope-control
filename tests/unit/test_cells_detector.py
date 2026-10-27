"""The cell detector: where the weights are, what is refused, and what torch's absence says (m3 §9.1).

Unless a test is marked *torch*, it runs without torch, so CI runs it: the
plugin must import and list on a PC that never installed the ``detect`` extra,
and a missing torch, a torch that does not load and a bad checkpoint must each
say what to do. The *torch* tests build a checkpoint of a few tensors with
``torch.save`` and are skipped without torch. The one test of the trained model
is skipped unless the weights are on the PC; CI never has them.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import smc
from smc.plugins import PluginRunError
from smc.plugins.cells import detector as detector_mod
from smc.plugins.cells.detector import (
    DEFAULT_MODEL,
    Detection,
    Detector,
    default_model_path,
    detections_from_output,
    load_detector,
    normalize_plane,
)

CHECKOUT = Path(smc.__file__).resolve().parents[2]
PREDICTOR = "roi_heads.box_predictor.cls_score.weight"


def _load(path: Path, **kwargs: Any) -> Detector:
    return load_detector(path, device="cpu", score_threshold=0.5, **kwargs)


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "w.pth"
    path.write_bytes(b"not a checkpoint")
    return path


def _reason(path: Path) -> str:
    with pytest.raises(PluginRunError) as caught:
        _load(path)
    return caught.value.reason


# --- importing without torch --------------------------------------------------


def test_the_plugin_imports_without_torch() -> None:
    """``smc run --list`` shows ``cells`` on every PC, torch or not."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["torch"] = None
        sys.modules["torchvision"] = None
        import smc.plugins.cells.plugin
        from smc.plugins.registry import load
        print(load("cells").plugin.provides)
        """
    )

    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert run.returncode == 0, run.stderr
    assert "cells" in run.stdout


# --- load_detector: what each failure says ------------------------------------


def test_a_missing_model_says_where_it_looked(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pth"

    reason = _reason(missing)

    assert str(missing) in reason
    assert "SMC_MODELS" in reason
    assert DEFAULT_MODEL in reason


def test_a_missing_torch_says_to_install_the_detect_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    reason = _reason(_weights(tmp_path))

    assert "torch is not installed" in reason
    assert "pip install -e .[detect]" in reason


def test_a_torch_that_does_not_load_is_told_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows raises ``OSError`` for a DLL that does not load: reinstalling is the cure."""

    class BrokenTorch:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("[WinError 126] The specified module could not be found")

    monkeypatch.setattr(detector_mod, "TorchDetector", BrokenTorch)

    reason = _reason(_weights(tmp_path))

    assert "did not load" in reason
    assert "WinError 126" in reason
    assert "not installed" not in reason


def test_a_refused_checkpoint_is_a_plugin_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ValueError("w.pth has no classes")

    monkeypatch.setattr(detector_mod, "TorchDetector", Refusing)

    assert _reason(_weights(tmp_path)) == "w.pth has no classes"


def test_the_cause_is_chained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Refusing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ValueError("why")

    monkeypatch.setattr(detector_mod, "TorchDetector", Refusing)

    with pytest.raises(PluginRunError) as caught:
        _load(_weights(tmp_path))

    assert isinstance(caught.value.__cause__, ValueError)


# --- where the weights are ----------------------------------------------------


def test_smc_models_sets_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMC_MODELS", str(tmp_path))

    assert default_model_path() == tmp_path / DEFAULT_MODEL


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_or_blank_smc_models_is_not_the_working_directory(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Path("")`` is ``.``: an empty variable must not make the cwd the models folder (FM-78)."""
    monkeypatch.setenv("SMC_MODELS", value)

    assert default_model_path() == CHECKOUT / "local" / "models" / DEFAULT_MODEL


def test_without_smc_models_the_weights_are_in_the_checkouts_local_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SMC_MODELS", raising=False)

    assert default_model_path() == CHECKOUT / "local" / "models" / DEFAULT_MODEL


# --- normalize_plane ----------------------------------------------------------


def test_normalize_plane_clips_the_percentiles_to_unit_range() -> None:
    ramp = np.tile(np.arange(100, dtype=np.uint16) * 10 + 500, (50, 1))
    ramp[0, 0] = 65535  # a hot pixel, which a min/max normalisation would obey

    out = normalize_plane(ramp)

    assert out.dtype == np.float32
    assert out.shape == ramp.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert out[0, 0] == 1.0
    # The ramp still spreads across the range: the hot pixel did not squash it.
    assert float(np.median(out)) == pytest.approx(0.5, abs=0.05)


def test_a_constant_frame_normalizes_to_zero() -> None:
    out = normalize_plane(np.full((8, 12), 1234, dtype=np.uint16))

    assert out.dtype == np.float32
    assert not out.any()


# --- the model's output -------------------------------------------------------


def test_the_output_boxes_change_columns_first_to_rows_first() -> None:
    """torchvision's ``[x0, y0, x1, y1]`` is ``(row0, col0, row1, col1)`` here (FM-73)."""
    boxes = np.array(
        [[10.0, 20.0, 30.0, 50.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.4, 0.5], dtype=np.float32)
    label_ids = np.array([2, 1, 3], dtype=np.int64)

    found = detections_from_output(
        boxes,
        scores,
        label_ids,
        labels=("__background__", "single", "doublet", "debris"),
        score_threshold=0.5,
    )

    assert [(d.row0_px, d.col0_px, d.row1_px, d.col1_px) for d in found] == [
        (20.0, 10.0, 50.0, 30.0),
        (6.0, 5.0, 8.0, 7.0),
    ]
    # The label id indexes the full list, background first; a score at the
    # threshold stays and one below it goes.
    assert [d.label for d in found] == ["doublet", "debris"]
    assert [d.score for d in found] == [pytest.approx(0.9), 0.5]
    assert all(isinstance(d, Detection) for d in found)


def test_an_output_without_a_box_is_no_detection() -> None:
    nothing = detections_from_output(
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.float32),
        np.zeros(0, np.int64),
        labels=("__background__", "single"),
        score_threshold=0.5,
    )

    assert nothing == []


# --- TorchDetector (torch only) -----------------------------------------------


def _checkpoint(tmp_path: Path, content: dict[str, Any]) -> Path:
    torch = pytest.importorskip("torch")
    path = tmp_path / "ckpt.pth"
    torch.save(content, path)
    return path


def _state(rows: int) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    # Linear(1024, rows): the right shape, so that only the missing keys can refuse it.
    return {PREDICTOR: torch.zeros(rows, 1024)}


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    path = tmp_path / "junk.pth"
    path.write_bytes(b"not a checkpoint")

    reason = _reason(path)

    assert "junk.pth" in reason
    assert "checkpoint" in reason


def test_a_checkpoint_that_is_not_a_dict_of_weights_is_refused(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, {"classes": ["__background__", "a"]})

    assert "model_state_dict" in _reason(path)


def test_a_checkpoint_without_classes_is_refused(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, {"model_state_dict": _state(4)})

    assert "classes" in _reason(path)


def test_a_checkpoint_whose_first_class_is_not_background_is_refused(
    tmp_path: Path,
) -> None:
    path = _checkpoint(
        tmp_path,
        {
            "model_state_dict": _state(4),
            "classes": ["single", "doublet", "debris", "other"],
        },
    )

    assert "__background__" in _reason(path)


def test_a_checkpoint_whose_predictor_rows_differ_from_its_classes_is_refused(
    tmp_path: Path,
) -> None:
    path = _checkpoint(
        tmp_path,
        {
            "model_state_dict": _state(3),
            "classes": ["__background__", "single", "doublet", "debris"],
        },
    )

    assert "3 outputs for 4 classes" in _reason(path)


def test_a_partial_state_is_refused_by_the_strict_load(tmp_path: Path) -> None:
    """Classes and rows agree, every other weight is missing: not the right architecture."""
    path = _checkpoint(
        tmp_path,
        {
            "model_state_dict": _state(4),
            "classes": ["__background__", "single", "doublet", "debris"],
        },
    )

    reason = _reason(path)

    assert "ckpt.pth" in reason
    assert "architecture" in reason


def test_an_unavailable_device_is_refused(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this PC has CUDA, so cuda is available here")
    path = _checkpoint(
        tmp_path,
        {
            "model_state_dict": _state(4),
            "classes": ["__background__", "single", "doublet", "debris"],
        },
    )

    with pytest.raises(PluginRunError) as caught:
        load_detector(path, device="cuda", score_threshold=0.5)

    assert caught.value.reason == (
        "device cuda is not available on this PC; use device=auto or cpu"
    )


# --- the trained model (torch and the weights) --------------------------------


def _trained() -> Detector:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    weights = default_model_path()
    if not weights.is_file():
        pytest.skip(f"no weights at {weights} (set SMC_MODELS to their folder)")
    return load_detector(None, device="auto", score_threshold=0.5)


def test_the_trained_model_loads_and_finds_nothing_on_noise() -> None:
    detector = _trained()

    assert detector.classes == ("single", "doublet", "debris")
    assert detector.device != "auto"
    noise = np.random.default_rng(0).normal(1000, 30, (300, 400)).astype(np.uint16)
    assert detector.detect(noise) == []


def test_the_trained_model_refuses_a_frame_that_is_not_two_dimensional() -> None:
    detector = _trained()

    with pytest.raises(ValueError, match="2-D"):
        detector.detect(np.zeros((40, 50, 3), np.uint8))
