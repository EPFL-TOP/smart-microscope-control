"""A scripted detector for tests: what a stage asks of a model, with no torch and no weights (inference.md §7).

``cells``' tests used to carry a stub each (``_Stub`` in the plugin's tests,
``_Scripted`` in ``cells_check``'s). One here is what they share, and what the
detector contract (``tests/contracts/test_detector.py``) runs against, so that
the protocol is tested once and the stub cannot drift from it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from smc.inference.detector import Detection, Detector
from smc.inference.models import ResolvedModel, model_identity
from smc.inference.preprocessing import ModelPreprocessing

__all__ = ["StubDetector", "stub_model"]


class StubDetector:
    """Returns what a test scripted for the frame it is given, keyed by its first pixel.

    A test writes frames whose first pixel says which one they are (``np.full`` of
    the frame's position), so that the stub knows which frame it was given and
    the test can say what that frame shows. ``seen`` lists the keys in the order
    the frames came; ``on_frame`` is called with each key before the detections
    are returned, which is where a test asks for a stop in the middle of a run.

    It does not check its labels against its ``classes``: ``cells``' refusal of a
    label its detector does not declare is tested with it.
    """

    classes: tuple[str, ...]
    device: str

    def __init__(
        self,
        scripted: Mapping[int, Sequence[Detection]] | None = None,
        *,
        default: Sequence[Detection] = (),
        classes: tuple[str, ...] = ("single", "doublet", "debris"),
        device: str = "cpu",
        on_frame: Callable[[int], None] | None = None,
    ) -> None:
        self._scripted = dict(scripted or {})
        self._default = tuple(default)
        self._on_frame = on_frame
        self.classes = classes
        self.device = device
        self.seen: list[int] = []

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The detections scripted for this frame's first pixel, or the default; a new list each time.

        Raises:
            ValueError: ``frame`` is not a 2-D array with at least one pixel.
        """
        array = np.asarray(frame)
        if array.ndim != 2 or array.size == 0:
            raise ValueError(
                f"detect takes one 2-D frame, got an array of shape {array.shape}"
            )
        key = int(array[0, 0])
        self.seen.append(key)
        if self._on_frame is not None:
            self._on_frame(key)
        return list(self._scripted.get(key, self._default))


def stub_model(
    detector: Detector,
    *,
    name: str = "stub",
    weights_sha256: str = "0" * 64,
    preprocessing: ModelPreprocessing | None = None,
) -> ResolvedModel:
    """A detector wrapped as a ``ResolvedModel``, for a stand-in of ``load_model``.

    The identity is worked out from the pre-processing it is given, as
    ``load_model`` does, so that a test whose stand-in receives a rule other than
    the one before it sees another identity and a resume refuses it (FM-260): a
    stand-in with a fixed identity would let the plugin ignore the rule.
    """
    return ResolvedModel(
        name=name,
        identity=model_identity(weights_sha256, preprocessing),
        weights_sha256=weights_sha256,
        runs_on=f"local:{detector.device}",
        detector=detector,
    )
