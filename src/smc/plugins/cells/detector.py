"""The cell detector: what the stage asks of one, and the trained model behind it (m3 §9.1).

``Detector`` is a protocol and there is no discovery in M3: one detector,
``TorchDetector``, and the stubs of the tests. The pure algorithm
(``algorithm.py``) only sees ``Detection``, so it runs on synthetic data with no
model.

Boxes are ``(row0, col0, row1, col1)`` in the pixels of the frame, as the cells
file stores them. torchvision emits ``[x0, y0, x1, y1]``; the swap is made once,
in ``TorchDetector.detect``, so nothing downstream thinks about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt

__all__ = ["Detection", "Detector"]


@dataclass(frozen=True, slots=True)
class Detection:
    """One box the detector found, in the pixels of its frame, with its label and score."""

    row0_px: float
    col0_px: float
    row1_px: float
    col1_px: float
    label: str
    score: float


@runtime_checkable
class Detector(Protocol):
    """What the cells stage asks of a detector: its labels, its device and ``detect``.

    ``classes`` are the labels it can return, background excluded; ``device`` is
    the one it runs on, resolved (never ``auto``), so the result can say what
    produced it.
    """

    classes: tuple[str, ...]
    device: str

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]:
        """The cells of one 2-D frame, in no particular order."""
        ...
