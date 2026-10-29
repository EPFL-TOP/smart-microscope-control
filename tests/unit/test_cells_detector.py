"""The cell detector: where the weights are, what is refused, and what torch's absence says (m3 §9.1).

Unless a test is marked *torch*, it runs without torch, so CI runs it: the
plugin must import and list on a PC that never installed the ``detect`` extra,
and a missing torch, a torch that does not load and a bad checkpoint must each
say what to do. The *torch* tests build a checkpoint of a few tensors with
``torch.save`` and are skipped without torch. The one test of the trained model
is skipped unless the weights are on the PC; CI never has them.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import smc
from smc.plugins import PluginRunError
from smc.plugins.cells import detector as detector_mod
from smc.plugins.cells.detector import (
    BLOB_DETECTOR,
    BLOB_DETECTOR_IDENTITY,
    DEFAULT_MODEL,
    BlobDetector,
    Detection,
    Detector,
    default_model_path,
    detections_from_output,
    load_detector,
    normalize_plane,
)
from smc.testing import Blob, PlateSample

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
    # torch is imported in the process that already holds the stand's DLLs (FM-95):
    # a fresh console tells that conflict from a broken install.
    assert "import torch" in reason


def test_a_refused_checkpoint_is_a_plugin_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ValueError("w.pth has no classes")

    monkeypatch.setattr(detector_mod, "TorchDetector", Refusing)

    assert _reason(_weights(tmp_path)) == "w.pth has no classes"


def test_a_model_path_the_os_refuses_to_stat_is_a_plugin_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder the user cannot enter, a share that dropped: ``is_file`` raises, it does not say no (FM-116)."""
    weights = tmp_path / DEFAULT_MODEL
    real = Path.is_file

    def refused(self: Path) -> bool:
        if self == weights:
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "is_file", refused)

    reason = _reason(weights)

    assert "Permission denied" in reason
    assert str(weights) in reason
    assert "SMC_MODELS" in reason


def test_an_import_error_that_is_not_a_missing_module_is_not_told_as_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DLL load failed while importing _imaging`` is an ``ImportError`` too."""

    class BrokenImport:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("DLL load failed while importing _imaging")

    monkeypatch.setattr(detector_mod, "TorchDetector", BrokenImport)

    reason = _reason(_weights(tmp_path))

    assert "DLL load failed while importing _imaging" in reason
    assert "could not be imported" in reason
    assert "not installed" not in reason
    assert "pip install -e .[detect]" in reason


def test_a_runtime_error_while_importing_torch_is_told_as_not_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torch and a torchvision of versions that do not go together raise ``RuntimeError``."""

    class Mismatched:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("operator torchvision::nms does not exist")

    monkeypatch.setattr(detector_mod, "TorchDetector", Mismatched)

    reason = _reason(_weights(tmp_path))

    assert "did not load" in reason
    assert "torchvision::nms" in reason


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

    with pytest.raises(PluginRunError) as caught:
        load_detector(path, device="cuda", score_threshold=0.5)

    assert caught.value.reason == (
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
) -> Detector:
    """A ``TorchDetector`` built by ``load_detector`` around ``model``, on a real small checkpoint."""
    import torchvision.models.detection as detection

    def build(**kwargs: Any) -> _FakeModel:
        model.built_with = kwargs
        return model

    monkeypatch.setattr(detection, "fasterrcnn_resnet50_fpn", build)
    path = _checkpoint(tmp_path, {"model_state_dict": _state(4), "classes": CLASSES})
    return load_detector(path, device=device, score_threshold=score_threshold)


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

    with pytest.raises(PluginRunError) as caught:
        _fake_detector(tmp_path, monkeypatch, model, device="auto")

    assert "no kernel image" in caught.value.reason
    assert "ckpt.pth" in caught.value.reason
    assert "device=cpu" in caught.value.reason


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

    with caplog.at_level(logging.WARNING, logger="smc.plugins.cells"):
        found = detector.detect(np.zeros((20, 30), np.uint16))

    assert len(found) == 2
    messages = [r.getMessage() for r in caplog.records if r.name == "smc.plugins.cells"]
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

    with caplog.at_level(logging.WARNING, logger="smc.plugins.cells"):
        detector.detect(np.zeros((20, 30), np.uint16))

    assert not [r for r in caplog.records if r.name == "smc.plugins.cells"]


# --- the trained model (torch and the weights) --------------------------------


def _trained() -> Detector:
    _torch()
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


# --- the blob detector, for the synthetic plate (m3 §9.1, #260) ----------------------

PIXEL_SIZES_UM = [0.1625, 0.25, 0.5, 1.0, 2.0]
#: The well's radius on the 96-well plate A1 sits at the stage origin of.
WELL_RADIUS_UM = 3200.0


def _spots(centres: list[tuple[float, float]]) -> list[Blob]:
    """Radius 5 um, 3000 counts: what ``smc.testing.sample_stand`` draws for a cell."""
    return [Blob(x, y, 5.0, 3000.0) for x, y in centres]


def _frame_with(blobs: list[Blob], pixel_size_um: float) -> np.ndarray[Any, Any]:
    sample = PlateSample(blobs=blobs, seed=3)
    return sample.render(0.0, 0.0, 0.0, shape=(512, 512), pixel_size_um=pixel_size_um)


def _labels(found: list[Detection]) -> Counter[str]:
    return Counter(d.label for d in found)


@pytest.mark.parametrize("pixel_size_um", PIXEL_SIZES_UM)
def test_the_blob_detector_finds_singles_and_doublets_at_every_pixel_size(
    pixel_size_um: float,
) -> None:
    w = 512 * pixel_size_um
    singles = [(-0.3 * w, -0.3 * w), (0.3 * w, -0.3 * w), (-0.3 * w, 0.3 * w)]
    singles += [(0.3 * w, 0.3 * w), (0.0, 0.0)]
    blobs = _spots(singles)
    # Three pairs along three angles, the two spots 4 um either side of a centre.
    for (cx, cy), angle in zip(
        [(0.0, 0.3 * w), (0.0, -0.3 * w), (-0.3 * w, 0.0)], [0.0, 1.0, 2.0], strict=True
    ):
        dx, dy = 4.0 * math.cos(angle), 4.0 * math.sin(angle)
        blobs += _spots([(cx - dx, cy - dy), (cx + dx, cy + dy)])

    found = BlobDetector().detect(_frame_with(blobs, pixel_size_um))

    assert _labels(found) == {"single": 5, "doublet": 3}
    assert {d.score for d in found} == {1.0}


@pytest.mark.parametrize("pixel_size_um", PIXEL_SIZES_UM)
def test_the_blob_detector_sees_nothing_across_a_well_wall(
    pixel_size_um: float,
) -> None:
    # A well wall is a step the background must not be mistaken for a cell by:
    # the frame is centred from half a frame inside the well to half outside, and
    # at the diagonal where the wall meets a corner.
    w = 512 * pixel_size_um
    centres = [
        (x, 0.0)
        for x in np.linspace(
            WELL_RADIUS_UM - 256 * pixel_size_um,
            WELL_RADIUS_UM + 256 * pixel_size_um,
            21,
        )
    ]
    centres.append((0.707 * WELL_RADIUS_UM + 0.2 * w, 0.707 * WELL_RADIUS_UM))
    sample = PlateSample(seed=3)

    for x, y in centres:
        frame = sample.render(x, y, 0.0, shape=(512, 512), pixel_size_um=pixel_size_um)
        assert BlobDetector().detect(frame) == [], (x, y)


@pytest.mark.parametrize("pixel_size_um", PIXEL_SIZES_UM)
def test_a_cell_in_the_sliver_of_well_beside_a_wall_is_still_found(
    pixel_size_um: float,
) -> None:
    # 40 % of the frame is well and 60 % plastic. The 75th percentile is then a
    # well pixel, so the sliver is background and the cell stands out of it; a
    # median would sit on the plastic, the whole sliver would be one bright
    # component (over 5 % of the frame, dropped) and the cell in it with it.
    w = 512 * pixel_size_um
    frame_x = WELL_RADIUS_UM + 0.1 * w
    cell = Blob(WELL_RADIUS_UM - 0.2 * w, 0.0, 5.0, 3000.0)
    sample = PlateSample(blobs=[cell], seed=3)
    frame = sample.render(
        frame_x, 0.0, 0.0, shape=(512, 512), pixel_size_um=pixel_size_um
    )

    found = BlobDetector().detect(frame)

    assert _labels(found) == {"single": 1}


def test_a_strip_is_not_a_cell() -> None:
    # A 3 x 60 px bar is bright, 180 px (inside the area limits), and not round:
    # only the axis ratio drops it.
    frame = np.random.default_rng(0).normal(1000, 20, (200, 200))
    frame[100:103, 70:130] += 3000

    assert BlobDetector().detect(frame) == []


def test_a_round_spot_is_a_single_and_an_elongated_one_a_doublet() -> None:
    rows, cols = np.mgrid[0:200, 0:300]
    frame = np.random.default_rng(1).normal(1000, 20, (200, 300))
    frame += 3000 * (((rows - 60) / 8.0) ** 2 + ((cols - 70) / 8.0) ** 2 <= 1)
    # Semi-axes 14 and 7: the minor/major ratio is 0.5, between 0.3 and 0.75.
    frame += 3000 * (((rows - 140) / 7.0) ** 2 + ((cols - 200) / 14.0) ** 2 <= 1)

    found = BlobDetector().detect(frame)

    assert sorted((round((d.row0_px + d.row1_px) / 2), d.label) for d in found) == [
        (60, "single"),
        (140, "doublet"),
    ]


def test_the_box_is_rows_then_columns_of_the_component() -> None:
    # A frame that is not square, and a blob that is not a square: a box with
    # the axes exchanged is wrong here and invisible on a square one (FM-73).
    frame = np.random.default_rng(2).normal(1000, 20, (100, 160))
    frame[20:30, 40:52] += 3000

    (box,) = BlobDetector().detect(frame)

    assert (box.row0_px, box.col0_px, box.row1_px, box.col1_px) == (20, 40, 30, 52)
    assert box.label == "single"


@pytest.mark.parametrize(
    ("side", "kept"), [(2, False), (3, True), (4, True)], ids=["4px", "9px", "16px"]
)
def test_a_component_under_nine_pixels_is_dropped(side: int, kept: bool) -> None:
    frame = np.random.default_rng(3).normal(1000, 20, (100, 100))
    frame[40 : 40 + side, 40 : 40 + side] += 3000

    assert (len(BlobDetector().detect(frame)) == 1) is kept


def test_a_component_over_five_percent_of_the_frame_is_dropped() -> None:
    # 100 x 100 is 10 000 px: a 22 x 22 blob (484 px) is under 5 % (500 px), a
    # 23 x 23 one (529 px) over it.
    frame = np.random.default_rng(4).normal(1000, 20, (100, 100))
    small = frame.copy()
    small[10:32, 10:32] += 3000
    large = frame.copy()
    large[10:33, 10:33] += 3000

    assert len(BlobDetector().detect(small)) == 1
    assert BlobDetector().detect(large) == []


def test_the_blob_detector_refuses_a_frame_that_is_not_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        BlobDetector().detect(np.zeros((40, 50, 3), np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        BlobDetector().detect(np.zeros(40, np.uint8))


@pytest.mark.parametrize("shape", [(0, 0), (5, 0), (0, 5), (5, 1), (1, 1)])
def test_a_frame_too_small_to_measure_gives_no_detection(
    shape: tuple[int, int],
) -> None:
    assert BlobDetector().detect(np.zeros(shape, np.uint16)) == []


@pytest.mark.parametrize("level", [0, 800, 65535])
def test_a_constant_frame_gives_no_detection(level: int) -> None:
    assert BlobDetector().detect(np.full((64, 80), level, np.uint16)) == []


def test_the_blob_detector_declares_its_classes_and_its_device() -> None:
    detector = BlobDetector()

    assert isinstance(detector, Detector)
    assert detector.classes == ("single", "doublet")
    assert detector.device == "cpu"


def test_the_blob_detectors_identity_is_the_one_the_design_names() -> None:
    # The resume compares the SHA-256 of this string: a change of a rule must
    # change it, and the string is part of the record, so it is pinned.
    assert BLOB_DETECTOR == "blobs"
    assert BLOB_DETECTOR_IDENTITY == (
        "smc BlobDetector 1: p75 + 8 sigma, 9 px to 5 % of the frame, "
        "axis ratio >= 0.3, single from 0.75"
    )
