"""The drawing primitives of the report: numbers, escaping, colours and the subsample (m3 §13).

Pure and stdlib only. What each one decides, because the page is opened on a
Windows PC, offline, and attached to issues:

* **Every number that reaches the page goes through ``num``.** A coordinate is
  printed to a tenth of a µm, and a value that is not finite, or so large that
  ``f"{1e300:.1f}"`` would print 303 characters, is refused naming the stage
  that holds it (FM-114), instead of becoming ``nan`` in an attribute that a
  browser silently drops.
* **Every free text goes through ``escape``** (markup built from a result's
  text). Nothing here is an ``id``, a class or a style: the colours are
  constants chosen by an index.
* **The subsample is exact and deterministic**: the same records give the same
  page, and a cap is a cap.
"""

from __future__ import annotations

import html
import math

__all__ = [
    "LABEL_COLOURS",
    "MAX_CELLS_DRAWN_TOTAL",
    "MAX_COORDINATE_UM",
    "density_text",
    "escape",
    "label_colour",
    "num",
    "sample_indices",
    "shade",
    "text_colour_on",
]

#: The most dots the whole page draws: a cap per well is no bound on the page
#: (20 000 cells in each of 96 wells is 1.9 million dots), and 300 000 dots are 4.7 MB.
MAX_CELLS_DRAWN_TOTAL = 300_000

#: Okabe-Ito, by the label's index in the density's labels, cycling: a legend
#: and a dot agree, and the set stays readable with the common colour-vision
#: deficiencies.
LABEL_COLOURS = (
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7A7A7A",
)

#: A coordinate beyond this is not a position on a plate: 1e12 µm is a million
#: km, and ``f"{1e300:.1f}"`` is 303 characters.
MAX_COORDINATE_UM = 1e12

#: Below this relative luminance white text has the larger WCAG contrast; above
#: it black has. The two contrasts cross at 4.58:1, so every step of every scale
#: reads at 4.5:1 or better.
_LUMINANCE_SPLIT = 0.179


def escape(text: object) -> str:
    """``text`` as HTML text and as a quoted attribute value: markup and both quotes are escaped."""
    return html.escape(str(text), quote=True)


def num(value: float, *, what: str, stage: str) -> str:
    """``value`` to a tenth, without a trailing ``.0`` and never as ``-0``.

    ``what`` names where the number comes from and ``stage`` the stage to run
    again, so that a refusal reaches the person who can fix it (``the plate's A1
    centre`` is the scan plan's).

    Raises:
        ValueError: ``value`` is not finite, or is beyond ``MAX_COORDINATE_UM``.
            The message ends ``; run <stage> again``.
    """
    if not math.isfinite(value):
        raise ValueError(f"{what} is not a finite number; run {stage} again")
    if abs(value) > MAX_COORDINATE_UM:
        raise ValueError(f"{what} is beyond 1e12 µm; run {stage} again")
    text = f"{value:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    # A tenth rounds -0.04 to -0.0, which is 0 and reads as a sign that is not there.
    return "0" if text == "-0" else text


def label_colour(index: int) -> str:
    """The colour of the label at ``index`` of the density's labels; the set cycles."""
    return LABEL_COLOURS[index % len(LABEL_COLOURS)]


def shade(colour: str, t: float) -> str:
    """The step of ``t`` on the scale from white to ``colour``, as lowercase ``#rrggbb``.

    ``mix = 0.10 + 0.90 * t`` with ``t`` clamped to ``[0, 1]``, so that a well
    of density 0 is paler than any other but not white: an unscanned well has no
    fill, and the two must differ. A ``t`` that is not finite is 0.
    """
    clamped = min(1.0, max(0.0, t)) if math.isfinite(t) else 0.0
    mix = 0.10 + 0.90 * clamped
    return "#" + "".join(
        f"{round(255 + (channel - 255) * mix):02x}" for channel in _channels(colour)
    )


def text_colour_on(fill: str) -> str:
    """Black or white, whichever has the larger WCAG contrast with ``fill``."""
    return "#000000" if _relative_luminance(fill) > _LUMINANCE_SPLIT else "#FFFFFF"


def sample_indices(n: int, cap: int) -> list[int]:
    """Which of ``n`` items to draw when at most ``cap`` can be: all, or an even, exact ``cap``.

    ``(i * n) // cap`` for ``i`` in ``range(cap)``: distinct, increasing and
    exactly ``cap`` of them, so the frames of a well are thinned evenly and the
    same records give the same page.

    Raises:
        ValueError: ``n`` is negative.
    """
    if n < 0:
        raise ValueError(f"cannot draw a sample of {n} items")
    if cap <= 0:
        return []
    if n <= cap:
        return list(range(n))
    return [(i * n) // cap for i in range(cap)]


def density_text(value: float) -> str:
    """A density per mm² with no exponent: whole from 100, one decimal from 10, two below.

    A density of exactly 0 is written ``0``, as the table writes a measured
    zero, not ``0.00``.
    """
    if value == 0.0:
        return "0"
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _channels(colour: str) -> tuple[int, int, int]:
    """The red, green and blue of a ``#rrggbb`` colour."""
    return (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16))


def _relative_luminance(colour: str) -> float:
    """The WCAG 2 relative luminance of a ``#rrggbb`` colour."""

    def linear(channel: int) -> float:
        c = channel / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in _channels(colour))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue
