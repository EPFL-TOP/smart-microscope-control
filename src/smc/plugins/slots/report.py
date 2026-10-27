"""The report slot: the static report of a scan (m3 §3.5)."""

from __future__ import annotations

from smc.plugins.api import PluginResult

__all__ = ["ReportResult"]


class ReportResult(PluginResult):
    """The report that was written; ``file`` is relative to the run folder, POSIX."""

    file: str
