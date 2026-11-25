"""The focus on the cells' slot: a sweep, a field's focus step, and the tool's result (m3 §21.5).

A sweep keeps its readbacks and scores, so a scan's record says why a frame
was taken where it was without keeping the sweep's frames. The tool keeps
them too, as files, and its result names them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from smc.plugins.api import PluginResult
from smc.plugins.cell_focus.algorithm import PeakOutcome, ScoreName
from smc.plugins.slots.base import SlotValue

__all__ = [
    "AnchorTry",
    "CellFocusResult",
    "FieldFocus",
    "FocusAnchor",
    "FocusMapUse",
    "FocusSweep",
]


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
    """A scan field's focus step: its sweep, the well's offset or the map, and what it cost.

    ``source`` is ``swept`` when the field was swept, ``well`` when it reused
    the ``dz_um`` its well's earlier field found (``per-well``, D5), ``map``
    when its Z came from the 10x lock map plus its well's anchor (m3 §21.12):
    then ``sweep`` is the check sweep around that Z (``map_check_every``), or
    ``None``, and the frame's Z command is ``map_z_um + anchor_um``.
    ``dz_um`` is the frame's Z command minus the centre, 0.0 when no peak
    was found and for ``map``; ``frame_score`` is the written frame's score,
    by the sweep's score; ``duration_s`` runs from before the switch-off (or
    the map's Z move) to after the snap.
    """

    source: Literal["swept", "well", "map"]
    sweep: FocusSweep | None
    dz_um: float
    frame_score: float
    duration_s: float = Field(ge=0.0)
    #: ``map``: the map's Z at the field's XY readback; ``None`` otherwise.
    map_z_um: float | None = None
    #: ``map``: its well's anchor; ``None`` otherwise.
    anchor_um: float | None = None

    @model_validator(mode="after")
    def _each_source_has_its_fields(self) -> FieldFocus:
        if (self.source == "swept" and self.sweep is None) or (
            self.source == "well" and self.sweep is not None
        ):
            raise ValueError(
                f"sweep is set when source is 'swept' and None when 'well', got "
                f"source={self.source!r} and sweep "
                f"{'set' if self.sweep is not None else 'None'}"
            )
        is_map = self.source == "map"
        if (self.map_z_um is not None) != is_map or (self.anchor_um is not None) != (
            is_map
        ):
            raise ValueError(
                f"map_z_um and anchor_um are set exactly when source is 'map', got "
                f"source={self.source!r}, map_z_um={self.map_z_um!r} and "
                f"anchor_um={self.anchor_um!r}"
            )
        return self

    @model_serializer(mode="wrap")
    def _leave_out_the_map_when_unused(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        # A swept or well focus is written exactly as before the map existed, so
        # an older smc still reads the scan: its model refuses a key it does not
        # know (FM-107).
        data: dict[str, Any] = handler(self)
        if self.map_z_um is None:
            data.pop("map_z_um", None)
        if self.anchor_um is None:
            data.pop("anchor_um", None)
        return data


class AnchorTry(SlotValue):
    """One sweep near a well's centre that tried to anchor the map at the 40x (m3 §21.12.4).

    ``x_um``, ``y_um`` are the XY readback and ``map_z_um`` the map there, all
    ``None`` when the move failed. ``sweep`` is set when the try's sweep ended,
    ``error`` (``<Type>: <message>``) when the try failed: exactly one of them.
    """

    name: str
    x_um: float | None
    y_um: float | None
    map_z_um: float | None
    sweep: FocusSweep | None
    error: str | None

    @model_validator(mode="after")
    def _a_sweep_or_an_error(self) -> AnchorTry:
        if (self.sweep is None) == (self.error is None):
            raise ValueError(
                "an anchor try holds exactly one of sweep and error, got "
                f"{'both' if self.sweep is not None else 'neither'}"
            )
        given = [v is not None for v in (self.x_um, self.y_um, self.map_z_um)]
        if any(given) and not all(given):
            raise ValueError(
                "x_um, y_um and map_z_um go together: all set or all None, got "
                f"{self.x_um!r}, {self.y_um!r} and {self.map_z_um!r}"
            )
        return self


class FocusAnchor(SlotValue):
    """A well's anchor: the offset added to the map's Z at every field of the well.

    ``swept``: the median of the peaks' offsets from the map, over the tries
    near the well's centre; ``start``: no try found a peak, and the operator's
    focus at the start anchors the well.
    """

    well: str
    source: Literal["swept", "start"]
    anchor_um: float
    tries: tuple[AnchorTry, ...]
    duration_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _swept_has_a_peak(self) -> FocusAnchor:
        peaks = sum(
            t.sweep is not None and t.sweep.outcome == "peak" for t in self.tries
        )
        if (peaks > 0) != (self.source == "swept"):
            raise ValueError(
                f"an anchor of source 'swept' holds a try with a peak and one of "
                f"'start' none, got source={self.source!r} with {peaks} peaks"
            )
        return self


class FocusMapUse(SlotValue):
    """Which map a scan's focus came from, the operator's anchor and each well's (m3 §21.12.4).

    ``run_doc`` is the path as typed, forward slashes, never resolved (FM-24,
    FM-50); ``measured`` the map's ``finished_utc`` to the second, or ``at an
    unknown time``; ``start_anchor_um`` is ``z_start`` minus the map at the
    start XY, of the attempt that started the scan (a resume keeps it: a stop
    leaves Z on a sweep plane, not at the operator's focus); ``anchors`` are in
    scan order, a resumed attempt's first.
    """

    run_doc: str
    scan_attempt: str
    measured: str
    start_anchor_um: float
    anchors: tuple[FocusAnchor, ...]


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
