"""``stage-report`` reads where the stage is and sends no move (m3 §1.11).

It exists so that the whole plugin path is exercised without a move: the entry
point, the parameters from text, the capability check, progress, stop, the run
document and the command line all run on the demo devices, and on a stand it
reads positions and sends nothing. Z is read when the stand has a focus drive
and the operator wants it; without one ``z_um`` is ``None``, unknown and not
zero ("unknown beats guessed").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from smc.hardware.capabilities import XYStage, ZStage
from smc.plugins.api import PluginParams, PluginResult, RunContext

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["StageReading", "StageReport", "StageReportParams", "StageReportResult"]


class StageReportParams(PluginParams):
    """How many readings, how far apart, and whether Z is wanted."""

    samples: int = Field(1, ge=1, le=100, description="How many readings to take.")
    interval_ms: float = Field(
        0.0, ge=0.0, le=10_000.0, description="Wait between two readings."
    )
    include_z: bool = Field(
        True, description="Also read Z when there is a focus stage."
    )


class StageReading(BaseModel):
    """One reading of the stage, in µm; ``z_um`` is ``None`` when Z was not read."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    x_um: float
    y_um: float
    z_um: float | None


class StageReportResult(PluginResult):
    """The profile the readings were taken on, and the readings in order."""

    profile: str
    readings: tuple[StageReading, ...]


class StageReport:
    """Reads the XY stage (and Z, when present and wanted) ``samples`` times."""

    name = "stage-report"
    summary = "Read where the stage is, and send no move."
    Params = StageReportParams
    Result = StageReportResult
    requires: tuple[type, ...] = (XYStage,)
    provides = "stage-report"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self, microscope: Microscope, params: StageReportParams, ctx: RunContext
    ) -> StageReportResult:
        """Read ``samples`` times, stopping between two readings when asked.

        The stop is checked before each reading and the wait is the context's
        (``ctx.wait_s``), which ends at a stop and, on Windows, leaves Ctrl-C
        deliverable (FM-83); there is no wait after the last reading.
        """
        stage = microscope.require(XYStage)
        focus = microscope.get(ZStage) if params.include_z else None
        readings: list[StageReading] = []
        for sample in range(params.samples):
            ctx.raise_if_stopped()
            xy = stage.position_um()
            z_um = focus.position_um() if focus is not None else None
            readings.append(StageReading(x_um=xy.x_um, y_um=xy.y_um, z_um=z_um))
            ctx.progress(sample + 1, params.samples, "read")
            if sample < params.samples - 1:
                ctx.wait_s(params.interval_ms / 1000)
        return StageReportResult(
            profile=microscope.profile.microscope.name, readings=tuple(readings)
        )
