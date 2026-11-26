"""The local detector: the pre-processing, the model's output as detections, and what a bad checkpoint says (m3 §9.1, §9.4).

Unless a test is marked *torch*, it runs without torch, so CI runs it: the
plugin must import and list on a PC that never installed the ``detect`` extra.
The *torch* tests build a checkpoint of a few tensors with ``torch.save`` and
are skipped without torch. The one test of the trained model is skipped unless
the weights are on the PC; CI never has them. Where the weights are looked for,
and what a missing torch says, is ``test_inference_models.py``'s.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from smc.inference import local as local_mod
from smc.inference.cards import DEFAULT_MODEL_NAME, model_card
from smc.inference.detector import Detection, Detector
from smc.inference.errors import ModelError
from smc.inference.local import detections_from_output, load_local
from smc.inference.models import weights_folder
from smc.inference.preprocessing import (
    DEFAULT_PREPROCESSING,
    MAX_INPUT_PX,
    MIN_INPUT_PX,
    Preprocessing,
    model_input_size_px,
    normalize_plane,
)

PREDICTOR = "roi_heads.box_predictor.cls_score.weight"


def _load(
    path: Path,
    *,
    preprocessing: Preprocessing = DEFAULT_PREPROCESSING,
    device: str = "cpu",
    score_threshold: float = 0.5,
) -> Detector:
    return load_local(
        path,
        device=device,
        score_threshold=score_threshold,
        preprocessing=preprocessing,
    )


def _reason(path: Path) -> str:
    with pytest.raises(ModelError) as caught:
        _load(path)
    return str(caught.value)


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


def test_the_bit_depth_does_not_change_the_normalised_plane() -> None:
    """A 12-bit frame and the same frame times 16 reach the model alike: no bit-depth parameter (m3 §9.4)."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 4096, size=(60, 80), dtype=np.uint16)

    low = normalize_plane(frame)
    high = normalize_plane(frame * np.uint16(16))

    assert float(np.abs(low - high).max()) <= 1e-6
    # And the frame is not flat, so the equality is not that of two blank planes.
    assert float(low.max()) == 1.0


def test_the_percentiles_asked_for_clip_the_plane() -> None:
    ramp = np.tile(np.arange(101, dtype=np.uint16), (4, 1))

    wide = normalize_plane(ramp, (0.0, 100.0))
    narrow = normalize_plane(ramp, (10.0, 90.0))

    # Percentile 10 of 0..100 is 10: the pixel at 10 is the floor, and 95 is above 90.
    assert float(narrow[0, 10]) == 0.0
    assert float(narrow[0, 95]) == 1.0
    assert 0.0 < float(wide[0, 10]) < 0.2


# --- pre-processing: the input size and the defaults (m3 §9.4, #342) -------------


def test_the_default_preprocessing_is_what_the_model_was_trained_with() -> None:
    assert Preprocessing() == Preprocessing(percentiles=(0.5, 99.5), input_size_px=None)


def test_model_input_size_follows_the_two_pixel_sizes() -> None:
    # 0.3 um per pixel shown at 0.6: half the pixels. Rows then columns, so the
    # shorter side first whichever way the frame is oriented.
    assert model_input_size_px((1200, 1600), 0.3, 0.6) == (600, 800)
    assert model_input_size_px((1600, 1200), 0.3, 0.6) == (600, 800)
    # round on the short side, ceil on the long one: 794.2 and 794.2.
    assert model_input_size_px((1412, 1412), 0.2756, 0.49) == (794, 795)


def test_the_long_side_never_binds_by_a_pixel() -> None:
    """torchvision scales by the smaller of min/short and max/long: a rounded-down long side would bind."""
    # 499.6 rounds up to 500 and 501.1 up to 502: a rounded long side (501) would give
    # a scale of 501 / 1003 = 0.4995, under 500 / 1000, and the long side would bind.
    short, long = model_input_size_px((1000, 1003), 0.4996, 1.0)

    assert (short, long) == (500, 502)
    assert short / 1000 <= long / 1003


@pytest.mark.parametrize(
    ("model_um", "size"), [(2.0, "195"), (0.09, "4324")], ids=["too small", "too large"]
)
def test_a_model_input_out_of_bounds_is_refused(model_um: float, size: str) -> None:
    assert (MIN_INPUT_PX, MAX_INPUT_PX) == (256, 4096)

    with pytest.raises(ValueError, match="would be") as caught:
        model_input_size_px((1412, 1412), 0.2756, model_um)

    message = str(caught.value)
    assert "0.2756" in message
    assert f"{model_um:g}" in message
    assert size in message
    assert message.isascii()


def test_the_bounds_are_inclusive() -> None:
    # 256 px is the smallest short side, 4096 the largest long side.
    assert model_input_size_px((512, 512), 0.5, 1.0) == (256, 256)
    assert model_input_size_px((2048, 4096), 1.0, 1.0)[1] == 4096
    with pytest.raises(ValueError, match="short side"):
        model_input_size_px((512, 510), 0.5, 1.0)
    with pytest.raises(ValueError, match="long side"):
        model_input_size_px((2048, 4097), 1.0, 1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -0.3])
def test_a_pixel_size_that_is_not_finite_and_positive_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="um_per_px"):
        model_input_size_px((1412, 1412), bad, 0.49)
    with pytest.raises(ValueError, match="um_per_px"):
        model_input_size_px((1412, 1412), 0.2756, bad)


def test_preprocessing_takes_the_whole_range_of_percentiles() -> None:
    """0 and 100 are the minimum and the maximum of the frame: a range, not an error."""
    assert Preprocessing(percentiles=(0.0, 100.0)).percentiles == (0.0, 100.0)
    assert Preprocessing(percentiles=(0.0, 0.1)).percentiles == (0.0, 0.1)
    assert Preprocessing(percentiles=(99.9, 100.0)).percentiles == (99.9, 100.0)


def test_preprocessing_refuses_percentiles_that_do_not_order() -> None:
    for bad in [(99.5, 0.5), (50.0, 50.0), (-1.0, 99.0), (1.0, 101.0)]:
        with pytest.raises(ValueError, match="percentiles"):
            Preprocessing(percentiles=bad)
    with pytest.raises(ValueError, match="percentiles"):
        Preprocessing(percentiles=(math.nan, 99.5))


def test_preprocessing_refuses_an_input_size_that_is_not_two_positive_sizes() -> None:
    with pytest.raises(ValueError, match="input_size_px"):
        Preprocessing(input_size_px=(0, 800))
    with pytest.raises(ValueError, match="input_size_px"):
        Preprocessing(input_size_px=(800, -1))


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


def _torch() -> Any:
    """torch, or a skip: ``TorchDetector`` imports torchvision as well, and both are the extra."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision.models.detection")
    return torch


def _checkpoint(tmp_path: Path, content: dict[str, Any]) -> Path:
    torch = _torch()
    path = tmp_path / "ckpt.pth"
    torch.save(content, path)
    return path


def _state(rows: int) -> dict[str, Any]:
    torch = _torch()
    # Linear(1024, rows): the right shape, so that only the missing keys can refuse it.
    return {PREDICTOR: torch.zeros(rows, 1024)}


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    _torch()
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
    torch = _torch()
    if torch.cuda.is_available():
        pytest.skip("this PC has CUDA, so cuda is available here")
    path = _checkpoint(
        tmp_path,
        {
            "model_state_dict": _state(4),
            "classes": ["__background__", "single", "doublet", "debris"],
        },
    )

    with pytest.raises(ModelError) as caught:
        _load(path, device="cuda")

    assert str(caught.value) == (
        "device cuda is not available on this PC; use device=auto or cpu"
    )


CLASSES = ["__background__", "single", "doublet", "debris"]


class _Payload:
    """Something a pickle can run code through: ``weights_only`` must refuse it."""


@pytest.mark.parametrize(
    "entry",
    [[1, 2, 3, 4], 5, "tensor", None],
    ids=["a list", "an int", "a string", "none"],
)
def test_a_predictor_entry_that_is_not_a_tensor_is_refused(
    tmp_path: Path, entry: object
) -> None:
    path = _checkpoint(
        tmp_path, {"model_state_dict": {PREDICTOR: entry}, "classes": CLASSES}
    )

    reason = _reason(path)

    assert "ckpt.pth" in reason
    assert "box_predictor" in reason


def test_a_zero_dimensional_predictor_entry_is_refused(tmp_path: Path) -> None:
    torch = _torch()
    path = _checkpoint(
        tmp_path,
        {"model_state_dict": {PREDICTOR: torch.tensor(5.0)}, "classes": CLASSES},
    )

    assert "box_predictor" in _reason(path)


def test_a_state_without_the_predictor_is_refused(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, {"model_state_dict": {}, "classes": CLASSES})

    reason = _reason(path)

    assert "ckpt.pth" in reason
    assert "box_predictor" in reason


def test_a_checkpoint_holding_a_pickled_object_is_refused(tmp_path: Path) -> None:
    """``weights_only=True`` is what makes the file safe to open: a pickle that runs code is refused."""
    path = _checkpoint(
        tmp_path,
        {
            "model_state_dict": _state(4),
            "classes": CLASSES,
            "extra": _Payload(),
        },
    )

    reason = _reason(path)

    assert "ckpt.pth" in reason
    assert "cannot be read as a checkpoint" in reason


class _FakeModel:
    """The Faster R-CNN's interface with a scripted output: what ``TorchDetector`` wires up.

    ``boxes`` are torchvision's ``[x0, y0, x1, y1]``. ``fail_to`` makes ``.to`` raise,
    as a CUDA that cannot run the model does.
    """

    def __init__(
        self,
        boxes: list[list[float]],
        scores: list[float],
        labels: list[int],
        *,
        cap: int = 100,
        fail_to: str | None = None,
    ) -> None:
        torch = _torch()
        self._output = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        self.roi_heads = SimpleNamespace(detections_per_img=cap)
        self._fail_to = fail_to
        self.strict: bool | None = None
        self.inputs: list[Any] = []
        self.built_with: dict[str, Any] = {}

    def load_state_dict(self, state: dict[str, Any], strict: bool) -> None:
        self.strict = strict

    def eval(self) -> _FakeModel:
        return self

    def to(self, device: str) -> _FakeModel:
        if device == self._fail_to:
            raise RuntimeError("CUDA error: no kernel image is available")
        return self

    def __call__(self, tensors: list[Any]) -> list[dict[str, Any]]:
        self.inputs = tensors
        return [self._output]


def _fake_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: _FakeModel,
    *,
    device: str = "cpu",
    score_threshold: float = 0.2,
    preprocessing: Preprocessing | None = None,
) -> Detector:
    """A ``TorchDetector`` built by ``load_local`` around ``model``, on a real small checkpoint."""
    import torchvision.models.detection as detection

    def build(**kwargs: Any) -> _FakeModel:
        model.built_with = kwargs
        return model

    monkeypatch.setattr(detection, "fasterrcnn_resnet50_fpn", build)
    path = _checkpoint(tmp_path, {"model_state_dict": _state(4), "classes": CLASSES})
    return _load(
        path,
        device=device,
        score_threshold=score_threshold,
        preprocessing=DEFAULT_PREPROCESSING if preprocessing is None else preprocessing,
    )


def test_detect_labels_thresholds_and_orders_what_the_model_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the wiring of ``detections_from_output``: which array is which, the labels, the threshold."""
    model = _FakeModel(
        boxes=[[10, 20, 30, 50], [1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]],
        scores=[0.9, 0.3, 0.7, 0.1],
        labels=[1, 2, 3, 1],
    )
    detector = _fake_detector(tmp_path, monkeypatch, model, score_threshold=0.2)

    found = detector.detect(np.arange(60 * 80, dtype=np.uint16).reshape(60, 80))

    assert [(d.row0_px, d.col0_px, d.row1_px, d.col1_px, d.label) for d in found] == [
        (20.0, 10.0, 50.0, 30.0, "single"),
        (2.0, 1.0, 4.0, 3.0, "doublet"),
        (6.0, 5.0, 8.0, 7.0, "debris"),
    ]
    assert [d.score for d in found] == [
        pytest.approx(0.9),
        pytest.approx(0.3),
        pytest.approx(0.7),
    ]
    assert detector.classes == ("single", "doublet", "debris")
    assert detector.device == "cpu"
    assert model.strict is True
    assert model.built_with["num_classes"] == 4
    # One frame, three channels of the frame's own size, scaled to [0, 1].
    (tensor,) = model.inputs
    assert tuple(tensor.shape) == (3, 60, 80)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_a_device_the_model_cannot_move_to_says_to_use_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = _FakeModel([], [], [], fail_to="cuda")

    with pytest.raises(ModelError) as caught:
        _fake_detector(tmp_path, monkeypatch, model, device="auto")

    assert "no kernel image" in str(caught.value)
    assert "ckpt.pth" in str(caught.value)
    assert "device=cpu" in str(caught.value)


def test_a_frame_that_fills_the_models_box_limit_is_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The model keeps its best 100 boxes per frame and says nothing: a denser frame is undercounted."""
    model = _FakeModel(
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4]],
        scores=[0.9, 0.8],
        labels=[1, 1],
        cap=2,
    )
    detector = _fake_detector(tmp_path, monkeypatch, model)

    with caplog.at_level(logging.WARNING, logger="smc.inference.local"):
        found = detector.detect(np.zeros((20, 30), np.uint16))

    assert len(found) == 2
    messages = [
        r.getMessage() for r in caplog.records if r.name == "smc.inference.local"
    ]
    assert len(messages) == 1
    assert "2" in messages[0]
    assert "undercounted" in messages[0]


def test_a_frame_below_the_models_box_limit_is_no_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _FakeModel(boxes=[[1, 1, 2, 2]], scores=[0.9], labels=[1], cap=2)
    detector = _fake_detector(tmp_path, monkeypatch, model)

    with caplog.at_level(logging.WARNING, logger="smc.inference.local"):
        detector.detect(np.zeros((20, 30), np.uint16))

    assert not [r for r in caplog.records if r.name == "smc.inference.local"]


def test_without_an_input_size_the_model_keeps_its_own_resize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``min_size`` or ``max_size`` is passed: torchvision's 800 and 1333, as trained."""
    model = _FakeModel([], [], [])
    _fake_detector(tmp_path, monkeypatch, model)

    assert "min_size" not in model.built_with
    assert "max_size" not in model.built_with


def test_the_model_input_size_reaches_the_resize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built by torchvision itself: its transform, not a fake, holds the sizes we asked for."""
    _torch()
    import torchvision.models.detection as detection

    real = detection.fasterrcnn_resnet50_fpn
    built: list[Any] = []

    def build(**kwargs: Any) -> Any:
        model = real(**kwargs)
        # The checkpoint of the helper is one tensor: what is under test is the resize.
        model.load_state_dict = lambda state, strict: None
        built.append(model)
        return model

    monkeypatch.setattr(detection, "fasterrcnn_resnet50_fpn", build)
    path = _checkpoint(tmp_path, {"model_state_dict": _state(4), "classes": CLASSES})

    _load(path, preprocessing=Preprocessing(input_size_px=(300, 400)))

    (model,) = built
    assert model.transform.min_size == (300,)
    assert model.transform.max_size == 400


def test_the_percentiles_reach_normalize_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam between the parameter and the plane the model sees (FM-124): a spy on the module's function."""
    seen: list[tuple[float, float]] = []
    real = local_mod.normalize_plane

    def spy(
        plane: np.ndarray[Any, Any], percentiles: tuple[float, float] = (0.5, 99.5)
    ) -> np.ndarray[Any, Any]:
        seen.append(percentiles)
        return real(plane, percentiles)

    monkeypatch.setattr(local_mod, "normalize_plane", spy)
    detector = _fake_detector(
        tmp_path,
        monkeypatch,
        _FakeModel([], [], []),
        preprocessing=Preprocessing(percentiles=(1.0, 99.0)),
    )

    detector.detect(np.arange(60 * 80, dtype=np.uint16).reshape(60, 80))

    assert seen == [(1.0, 99.0)]


def test_the_default_percentiles_are_the_trained_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[float, float]] = []
    real = local_mod.normalize_plane

    def spy(
        plane: np.ndarray[Any, Any], percentiles: tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray[Any, Any]:
        seen.append(percentiles)
        return real(plane, percentiles)

    monkeypatch.setattr(local_mod, "normalize_plane", spy)
    detector = _fake_detector(tmp_path, monkeypatch, _FakeModel([], [], []))

    detector.detect(np.arange(60 * 80, dtype=np.uint16).reshape(60, 80))

    assert seen == [(0.5, 99.5)]


# --- the trained model (torch and the weights) --------------------------------


def _trained() -> Detector:
    _torch()
    weights = weights_folder() / model_card(DEFAULT_MODEL_NAME).weights
    if not weights.is_file():
        pytest.skip(f"no weights at {weights} (set SMC_MODELS to their folder)")
    return _load(weights, device="auto")


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
