"""The focus on the cells' slot: a sweep, a field's focus step, and the tool's result (m3 §21.5).

A sweep keeps its readbacks and scores, so a scan's record says why a frame
was taken where it was without keeping the sweep's frames. The tool keeps
them too, as files, and its result names them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from smc.plugins.api import PluginResult
from smc.plugins.cell_focus.algorithm import PeakOutcome, ScoreName
from smc.plugins.slots.base import SlotValue

__all__ = ["CellFocusResult", "FieldFocus", "FocusSweep"]


class FocusSweep(SlotValue):
    """One sweep: its score, its centre, the Z read at each plane and each plane's score.

    ``z_um`` are the readbacks, increasing, not the commands: the peak is
    refined through them. ``best_z_um`` is the refined plane, set only when
    the sweep found a peak (D4); every other outcome kept the centre.
    """

    score: ScoreName
    centre_um: float
    z_um: tuple[float, ...]
    scores: tuple[float, ...]
    outcome: PeakOutcome
    best_z_um: float | None

    @model_validator(mode="after")
    def _a_sweep_holds_together(self) -> FocusSweep:
        if len(self.z_um) != len(self.scores):
            raise ValueError(
                f"z_um and scores must have the same length, got {len(self.z_um)} "
                f"and {len(self.scores)}"
            )
        if len(self.z_um) < 3:
            raise ValueError(
                f"z_um and scores need at least 3 planes, got {len(self.z_um)}"
            )
        if (self.best_z_um is not None) != (self.outcome == "peak"):
            raise ValueError(
                f"best_z_um is set exactly when outcome is 'peak', got outcome="
                f"{self.outcome!r} and best_z_um={self.best_z_um!r}"
            )
        return self


class FieldFocus(SlotValue):
    """A scan field's focus step: its sweep, or the well's offset, and what it cost.

    ``source`` is ``swept`` when the field was swept, ``well`` when it reused
    the ``dz_um`` its well's earlier field found (``per-well``, D5).
    ``dz_um`` is the frame's Z command minus the centre, 0.0 when no peak
    was found; ``frame_score`` is the written frame's score, by the sweep's
    score; ``duration_s`` runs from before the switch-off to after the snap.
    """

    source: Literal["swept", "well"]
    sweep: FocusSweep | None
    dz_um: float
    frame_score: float
    duration_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _swept_has_its_sweep(self) -> FieldFocus:
        if (self.sweep is not None) != (self.source == "swept"):
            raise ValueError(
                f"sweep is set exactly when source is 'swept', got source="
                f"{self.source!r} and sweep "
                f"{'set' if self.sweep is not None else 'None'}"
            )
        return self


class CellFocusResult(PluginResult):
    """What ``focus-on-cells`` did at one place: the sweep, its frames, and how it left the stand.

    ``folder`` is ``cell-focus/<token>``, relative to the run folder, POSIX;
    ``files`` one frame per plane in the sweep's order, empty in a dry run.
    ``autofocus`` is the state at the start (``locked`` when it was used);
    ``relocked`` is the relock's answer, ``None`` when the autofocus was not
    used or not relocked. ``z_um`` is the Z read at the end.
    """

    folder: str
    objective: str | None
    x_um: float | None
    y_um: float | None
    autofocus: Literal["locked", "off", "none"]
    pfs_offset: float | None
    sweep: FocusSweep
    files: tuple[str, ...]
    best_file: str | None
    z_um: float
    relocked: bool | None

    @model_validator(mode="after")
    def _one_file_per_plane(self) -> CellFocusResult:
        if self.files and len(self.files) != len(self.sweep.z_um):
            raise ValueError(
                f"files must be empty (a dry run) or one per plane: got "
                f"{len(self.files)} files for {len(self.sweep.z_um)} planes in sweep.z_um"
            )
        return self
