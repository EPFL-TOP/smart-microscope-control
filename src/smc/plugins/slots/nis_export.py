"""The NIS-Elements export slot: the route as a file for the Nikon software (m3 §3.5)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from smc.plugins.api import PluginResult

__all__ = ["NisExportResult"]


class NisExportResult(PluginResult):
    """The export that was written: its file, its positions and how Z was carried.

    ``file`` is relative to the run folder, POSIX. ``z_mode`` says what the
    export holds for Z: the stage's ``z``, the autofocus's ``pfs-offset``, or
    ``none``.
    """

    file: str
    positions: int = Field(ge=0)
    z_mode: Literal["z", "pfs-offset", "none"]
