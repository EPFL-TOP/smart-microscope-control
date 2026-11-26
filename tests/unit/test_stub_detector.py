"""The scripted detector of ``smc.testing``: what the cells tests rely on it for (inference.md §7).

The detector contract (``tests/contracts/test_detector.py``) says it is a
``Detector``; these say what is particular to it: it is keyed by the frame's
first pixel, and the identity of its model follows the rule it is given.
"""

from __future__ import annotations

import numpy as np
import pytest

from smc.inference.detector import Detection
from smc.inference.preprocessing import ModelPreprocessing
from smc.testing import StubDetector, stub_model

ONE = Detection(1.0, 2.0, 3.0, 4.0, "single", 0.9)
TWO = Detection(5.0, 6.0, 7.0, 8.0, "doublet", 0.8)


def _frame(key: int) -> np.ndarray:  # type: ignore[type-arg]
    return np.full((6, 8), key, dtype=np.uint16)


def test_it_returns_what_was_scripted_for_the_frames_first_pixel() -> None:
    stub = StubDetector({0: [ONE], 7: [TWO, ONE]})

    assert stub.detect(_frame(0)) == [ONE]
    assert stub.detect(_frame(7)) == [TWO, ONE]
    assert stub.seen == [0, 7]


def test_a_frame_nobody_scripted_gets_the_default() -> None:
    assert StubDetector({0: [ONE]}).detect(_frame(3)) == []
    assert StubDetector(default=[TWO]).detect(_frame(3)) == [TWO]


def test_it_returns_a_new_list_each_time() -> None:
    stub = StubDetector({0: [ONE]})

    first = stub.detect(_frame(0))
    first.append(TWO)

    assert stub.detect(_frame(0)) == [ONE]


def test_it_calls_on_frame_with_the_key_before_it_answers() -> None:
    events: list[int] = []
    stub = StubDetector({1: [ONE]}, on_frame=events.append)

    stub.detect(_frame(1))
    stub.detect(_frame(2))

    assert events == [1, 2]


def test_it_does_not_check_its_labels_against_its_classes() -> None:
    """``cells``' refusal of an undeclared label is tested with it."""
    alien = Detection(0.0, 0.0, 1.0, 1.0, "alien", 0.5)

    assert StubDetector({0: [alien]}).detect(_frame(0)) == [alien]


@pytest.mark.parametrize("shape", [(4, 4, 3), (4,), (0, 5)])
def test_it_refuses_a_frame_that_is_not_a_two_dimensional_one(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="2-D"):
        StubDetector().detect(np.zeros(shape, np.uint16))


def test_its_classes_and_device_are_the_ones_it_is_given() -> None:
    stub = StubDetector(classes=("a",), device="cuda")

    assert stub.classes == ("a",)
    assert stub.device == "cuda"


def test_the_identity_of_a_stub_model_follows_the_rule_it_is_given() -> None:
    """FM-260: a stand-in that ignored the rule would let the plugin ignore it too."""
    stub = StubDetector()
    native = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
    other = ModelPreprocessing(percentiles=(1.0, 99.5), model_um_per_px=None)

    a = stub_model(stub, name="m", preprocessing=native)
    b = stub_model(stub, name="m", preprocessing=other)

    assert a.identity != b.identity
    assert a.weights_sha256 == b.weights_sha256 == "0" * 64
    assert a.runs_on == "local:cpu"
    assert a.name == "m"
    assert a.detector is stub
