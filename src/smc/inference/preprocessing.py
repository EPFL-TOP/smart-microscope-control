"""What is done to a frame before a model sees it (m3 §9.4).

The model was trained on percentile-normalised planes, and a frame can be shown
to it at another scale than its own resize. ``Preprocessing`` is that as the
detector applies it; the rest of this module is the arithmetic behind it, numpy
and the standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, StrictFloat, field_validator

__all__ = [
    "DEFAULT_PREPROCESSING",
    "MAX_INPUT_PX",
    "MIN_INPUT_PX",
    "ModelPreprocessing",
    "Preprocessing",
    "model_input_size_px",
    "normalize_plane",
]

#: The largest pixel size, in µm, a card or a parameter may name: the stand's cameras
#: are under 1 µm per pixel at 10x and above, and a larger number is a typo.
_MAX_UM_PER_PX = 10.0


def pixel_size_or_word(value: object, word: str) -> object:
    """A card's pixel size: the number, or ``None`` for the word that says there is none.

    TOML has no null (measured, inference.md §1), so a card says *unknown* and
    *native* in words and reads them here, before the strict float check. Any
    other string is refused with what to write instead: ``StrictFloat`` alone
    would only say "not a number".
    """
    if isinstance(value, str):
        if value == word:
            return None
        raise ValueError(
            f"{value!r} is not a pixel size: write a number in um, or the word {word!r}"
        )
    return value


def check_pixel_size(value: float | None) -> float | None:
    """A pixel size above 0 and at most 10 um, or ``None``; a nan fails the test, not passes it."""
    if value is not None and not (0.0 < value <= _MAX_UM_PER_PX):
        raise ValueError(f"{value:g} um is not above 0 and at most {_MAX_UM_PER_PX:g}")
    return value


def normalize_plane(
    plane: npt.NDArray[Any], percentiles: tuple[float, float] = (0.5, 99.5)
) -> npt.NDArray[np.float32]:
    """A 2-D frame as float32 in [0, 1], clipped at two percentiles.

    The model was trained on percentile-normalised planes. Percentiles and not
    the minimum and maximum, so that a hot pixel or the bright rim of a well does
    not squash the cells into a few grey levels. A constant frame has no range to
    scale and comes out all 0 (the epsilon keeps the division finite).
    """
    values = np.asarray(plane, dtype=np.float32)
    low, high = np.percentile(values, percentiles)
    scaled = np.clip((values - low) / (high - low + 1e-6), 0.0, 1.0)
    return np.asarray(scaled, dtype=np.float32)


#: The smallest short side and the largest long side, in pixels, that the model is asked
#: to resize a frame to. Below 256 a cell of the stand's 40x frames is a few pixels and
#: the backbone has nothing to see; above 4096 a forward pass costs seconds and the
#: 100-box limit is reached on any frame with cells (m3 §9.4).
MIN_INPUT_PX = 256
MAX_INPUT_PX = 4096


@dataclass(frozen=True, slots=True)
class Preprocessing:
    """What is done to a frame before the model sees it, beyond what it was trained with.

    ``percentiles`` are the intensity scaling of ``normalize_plane``: as trained,
    0.5 and 99.5. ``input_size_px`` is torchvision's ``(min_size, max_size)``, the size
    the frame is resized to before the network, and ``None`` keeps the model's own 800
    and 1333. Its boxes come back in the frame's own pixels either way. The bit depth
    is not here: a gain changes nothing the percentiles do not undo (measured, §9.4).
    """

    percentiles: tuple[float, float] = (0.5, 99.5)
    input_size_px: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        low, high = self.percentiles
        # `not (a < b)` and not `a >= b`: nan fails the first and passes the second.
        if not (0.0 <= low < high <= 100.0):
            raise ValueError(
                f"percentiles must satisfy 0 <= low < high <= 100, got {self.percentiles}"
            )
        if self.input_size_px is not None and min(self.input_size_px) < 1:
            raise ValueError(
                f"input_size_px must be two sizes of at least 1 pixel, "
                f"got {self.input_size_px}"
            )


#: How the model was trained: its own resize and the 0.5 / 99.5 percentiles. The default
#: of the two constructors below, a frozen value so that sharing it is safe.
DEFAULT_PREPROCESSING = Preprocessing()


def model_input_size_px(
    image_shape_px: tuple[int, int],
    frame_um_per_px: float,
    model_um_per_px: float,
) -> tuple[int, int]:
    """The ``(min_size, max_size)`` that shows a frame to the model at ``model_um_per_px``.

    The model cannot be told a pixel size, only the size its input is resized to, so
    the scale is the ratio of the two pixel sizes. torchvision scales by the smaller
    of ``min_size / short side`` and ``max_size / long side``: the short side is
    rounded and the long side rounded **up**, so that the short side sets the scale
    and the long one cannot bind by a pixel. A frame of 1412 pixels at 0.2756 µm
    shown at 0.49 µm is ``(794, 795)``.

    Raises:
        ValueError: A pixel size is not finite and positive, or the size is under
            ``MIN_INPUT_PX`` on the short side or over ``MAX_INPUT_PX`` on the long
            one. The message names both pixel sizes and the size it would give.
    """
    for name, value in (
        ("frame_um_per_px", frame_um_per_px),
        ("model_um_per_px", model_um_per_px),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite pixel size above 0, got {value}")
    scale = frame_um_per_px / model_um_per_px
    short_px = min(image_shape_px) * scale
    long_px = max(image_shape_px) * scale
    if not (math.isfinite(short_px) and math.isfinite(long_px)):
        raise ValueError(
            f"frame_um_per_px {frame_um_per_px:g} over model_um_per_px "
            f"{model_um_per_px:g} is not a usable scale"
        )
    size = (round(short_px), math.ceil(long_px))
    if size[0] < MIN_INPUT_PX or size[1] > MAX_INPUT_PX:
        raise ValueError(
            f"a {image_shape_px[0]}x{image_shape_px[1]} px frame at {frame_um_per_px:g} "
            f"um per pixel shown to the model at {model_um_per_px:g} um per pixel would "
            f"be {size[0]}x{size[1]} px; the short side must be at least "
            f"{MIN_INPUT_PX} and the long side at most {MAX_INPUT_PX}"
        )
    return size


class ModelPreprocessing(BaseModel):
    """What is done to a frame before the model sees it, as a rule (the card's table).

    The rule is what a card names and what a run applies, so that it can be
    written down, digested into the model's identity and compared; the
    ``Preprocessing`` of the detector is that rule worked out for one scan's
    frames. Its bounds are those of ``cells``' ``low_percentile``,
    ``high_percentile`` and ``model_um_per_px``, so that a parameter that replaces
    one part of a card's rule always leaves a valid rule. ``model_um_per_px`` is
    ``None`` for the model's own resize ("native" in a card).
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, protected_namespaces=()
    )

    percentiles: tuple[StrictFloat, StrictFloat]
    model_um_per_px: StrictFloat | None

    @field_validator("percentiles")
    @classmethod
    def _percentiles_surround_the_median(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        low, high = value
        # `not (a < b)`: a nan fails the chain, which `a >= b` would let through.
        if not (0.0 <= low < 50.0 < high <= 100.0):
            raise ValueError(
                f"{list(value)} does not satisfy 0 <= low < 50 < high <= 100, the "
                f"bounds of cells' low_percentile and high_percentile"
            )
        return value

    @field_validator("model_um_per_px", mode="before")
    @classmethod
    def _native_is_none(cls, value: object) -> object:
        return pixel_size_or_word(value, "native")

    @field_validator("model_um_per_px")
    @classmethod
    def _model_pixel_size_is_plausible(cls, value: float | None) -> float | None:
        return check_pixel_size(value)
