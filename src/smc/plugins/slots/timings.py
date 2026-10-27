"""The timings slot: what a move, a refocus and a snap cost on this stand (m3 §3.5)."""

from __future__ import annotations

import math

from pydantic import Field

from smc.plugins.api import PluginResult

__all__ = ["TimingsResult"]


class TimingsResult(PluginResult):
    """The measured cost of the scan's elementary steps.

    ``refocus_ms`` is ``None`` on a stand without autofocus. A value that was
    assumed rather than measured is named in ``assumed``, as ``nikon-control``'s
    ``timing.py`` did.
    """

    move_overhead_ms: float = Field(ge=0.0)
    move_ms_per_mm: float = Field(ge=0.0)
    refocus_ms: float | None = Field(ge=0.0)
    snap_ms: float = Field(ge=0.0)
    # default_factory, not ``= {}``: RUF012 does not see through ``PluginResult``.
    channel_ms: dict[str, float] = Field(default_factory=dict)
    channel_switch_ms: float | None = Field(None, ge=0.0)
    samples: int = Field(ge=0)

    def move_ms(self, distance_um: float) -> float:
        """The time of a stage move of ``distance_um``, in ms: an overhead plus a linear part.

        A distance of 0 or less costs nothing (the stage does not move), the
        model of ``nikon-control``'s ``timing.py``.

        Raises:
            ValueError: ``distance_um`` is not finite (``nan`` would slip past
                the ``<= 0`` test and give a ``nan`` time).
        """
        if not math.isfinite(distance_um):
            raise ValueError(f"distance_um must be finite, got {distance_um!r}")
        if distance_um <= 0.0:
            return 0.0
        return self.move_overhead_ms + self.move_ms_per_mm * distance_um / 1000.0
