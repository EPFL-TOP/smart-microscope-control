"""The ``Detector`` contract, on every implementation (inference.md §7).

What a stage may rely on from any detector, whatever runs behind it: the stub of
the tests, the blob detector, the served client on a fake model server (#685), the
trained Faster R-CNN (a random-initialised one, built once per module, so that no
real weights are needed). The frame is 120 x 160, which is not square, so that a box
whose axes are exchanged is outside it (FM-73), and it has three round spots.

A random model finds boxes on any frame (100 on noise, measured): "nothing on a
blank frame" is not a contract, and no case here asks for it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from smc.inference.blobs import BlobDetector
from smc.inference.cards import ModelCard
from smc.inference.detector import Detection, Detector
from smc.inference.models import load_model
from smc.inference.preprocessing import ModelPreprocessing
from smc.testing import FakeModelServer, StubDetector, stub_model

HEIGHT, WIDTH = 120, 160
#: Rows and columns of the three discs, radius 4 px, at 3000 counts over N(1000, 30).
DISCS = [(30, 40), (60, 100), (95, 140)]
_BACKGROUND = "__background__"


def contract_frame(dtype: Any = np.uint16) -> npt.NDArray[Any]:
    """The frame every case looks at: noise and three discs, 120 rows by 160 columns."""
    frame = np.random.default_rng(0).normal(1000, 30, (HEIGHT, WIDTH))
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    for row, col in DISCS:
        frame += 3000 * ((rows - row) ** 2 + (cols - col) ** 2 <= 4**2)
    return np.asarray(frame.astype(dtype))


def _stub() -> Detector:
    return StubDetector(
        default=[
            Detection(row - 5.0, col - 5.0, row + 5.0, col + 5.0, "single", 0.9)
            for row, col in DISCS
        ]
    )


@pytest.fixture(scope="module")
def torch_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random-initialised Faster R-CNN with three classes, saved as the training does (166 MB)."""
    torch = pytest.importorskip("torch", reason="the detect extra is not installed")
    detection = pytest.importorskip(
        "torchvision.models.detection", reason="the detect extra is not installed"
    )
    torch.manual_seed(0)
    model = detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=4
    )
    path = tmp_path_factory.mktemp("detector-contract") / "random.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": [_BACKGROUND, "single", "doublet", "debris"],
        },
        path,
    )
    return path


def _served(monkeypatch: pytest.MonkeyPatch) -> Detector:
    """A ``ServedDetector`` on a ``FakeModelServer`` around the stub's three detections.

    Through ``load_model``, the way ``cells`` gets one: the variable is set, the card
    is a model in memory, and the frame goes through TIFF and JSON and comes back.
    """
    rule = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
    stub = _stub()
    card = ModelCard(
        name="served-stub",
        weights="stub.pth",
        weights_sha256="0" * 64,
        classes=stub.classes,
        trained_um_per_px=None,
        preprocessing=rule,
    )
    fake = FakeModelServer(
        {
            card.name: stub_model(
                stub,
                name=card.name,
                weights_sha256=card.weights_sha256,
                preprocessing=rule,
            )
        }
    )
    fake.install(monkeypatch)
    resolved = load_model(
        card,
        preprocessing=rule,
        frame_shape_px=(HEIGHT, WIDTH),
        frame_um_per_px=0.5,
        device="auto",
        score_threshold=0.05,
    )
    assert resolved.runs_on == "served"
    return resolved.detector


@pytest.fixture(params=["stub", "blobs", "served", "torch"])
def detector(request: pytest.FixtureRequest) -> Detector:
    """Each implementation of the protocol; the torch case is skipped with its reason."""
    if request.param == "stub":
        return _stub()
    if request.param == "blobs":
        return BlobDetector()
    if request.param == "served":
        return _served(request.getfixturevalue("monkeypatch"))
    from smc.inference.local import TorchDetector
    from smc.inference.preprocessing import Preprocessing

    weights: Path = request.getfixturevalue("torch_weights")
    return TorchDetector(
        weights,
        device="cpu",
        score_threshold=0.05,
        preprocessing=Preprocessing(),
    )


def test_it_is_a_detector_with_classes_and_a_resolved_device(
    detector: Detector,
) -> None:
    assert isinstance(detector, Detector)
    assert isinstance(detector.classes, tuple)
    assert detector.classes
    assert all(isinstance(label, str) and label for label in detector.classes)
    assert len(set(detector.classes)) == len(detector.classes)
    assert _BACKGROUND not in detector.classes
    # Resolved: the result says what produced it, and `auto` says nothing.
    assert isinstance(detector.device, str)
    assert detector.device
    assert detector.device != "auto"


def test_its_detections_are_in_the_frame_with_a_declared_label_and_a_unit_score(
    detector: Detector,
) -> None:
    found = detector.detect(contract_frame())

    assert found
    for d in found:
        assert isinstance(d, Detection)
        values = (d.row0_px, d.col0_px, d.row1_px, d.col1_px, d.score)
        assert all(math.isfinite(v) for v in values), d
        assert 0.0 <= d.row0_px <= d.row1_px <= HEIGHT, d
        assert 0.0 <= d.col0_px <= d.col1_px <= WIDTH, d
        assert d.label in detector.classes, d
        assert 0.0 <= d.score <= 1.0, d


@pytest.mark.parametrize(
    "shape", [(40, 50, 3), (40,), ()], ids=["colour", "one-dimensional", "scalar"]
)
def test_it_refuses_a_frame_that_is_not_two_dimensional(
    detector: Detector, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="2-D"):
        detector.detect(np.zeros(shape, np.uint16))


def test_the_same_frame_gives_the_same_detections(detector: Detector) -> None:
    frame = contract_frame()

    assert detector.detect(frame) == detector.detect(frame)


def test_it_leaves_the_frame_as_it_was(detector: Detector) -> None:
    frame = contract_frame()
    before = frame.copy()

    detector.detect(frame)

    assert frame.dtype == before.dtype
    assert np.array_equal(frame, before)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_it_takes_a_uint16_and_a_float32_frame(detector: Detector, dtype: Any) -> None:
    found = detector.detect(contract_frame(dtype))

    assert isinstance(found, list)
    assert all(d.label in detector.classes for d in found)
