"""``plate-finding`` surveys the plate with the stage and the camera, and fits which plate it is and where (m3 §6).

The pure search is ``algorithm.py``. This is the hardware loop around it, and
the decisions in it that keep a survey of hundreds of absolute moves from doing
harm or from lying:

* **Everything that can be checked is checked before anything moves** (FM-30):
  a dry run, the checkpoint, the survey's size, the stage's soft limits and
  whether what is left of the survey can pin A1 down at all. A refusal leaves
  the stage where it was and the autofocus as it was.
* **The autofocus is switched off and never engaged again** (FM-12, FM-88). It
  cannot lock over the plastic between wells, and engaging it drives Z. The
  result says what the scan will find: the autofocus's state, read at the end of
  the survey and not remembered from its start, so a resumed survey says it too.
* **Nothing moves to put something back** (FM-88): the stage stays at the last
  point, there is no return and no ``finally`` that moves. No Z, no turret.
* **A halt is not a skipped point** (FM-64). A point that fails is counted and
  skipped; ``max_failed_in_row`` of them in a row, or more than a tenth of the
  survey, end it, and the checkpoint keeps what was sampled. The bound in a row
  comes first: a tenth of 1120 points at the 60 s move timeout is 1.9 h of a
  dead stage.
* **A frame is placed where the stage says it was**, not where it was sent
  (measured beats assumed). A stage that stops short of a target without an
  error gives a survey that does not reach the plate's edges, and the extent
  checks refuse it.
* **A fit that cannot be trusted is not a result.** The extent is checked again
  after the fit, for the plate that won: a winner larger than the survey, or a
  survey with its edge skipped, can leave A1 a whole well off while the score is
  high.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import useq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smc.hardware.capabilities import Autofocus, Camera, XYStage
from smc.hardware.errors import HardwareError, MicroscopeHaltedError, MotionStoppedError
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.plate_finding.algorithm import (
    SurveyPoint,
    decide,
    fit,
    frame_value,
    survey_extent_note,
    survey_points,
)
from smc.plugins.slots import PlateResult

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["PlateFindingParams", "PlateFindingPlugin", "SurveyCheckpoint"]

_NAME = "plate-finding"
#: The checkpoint is saved every this many points, and at the end.
_CHECKPOINT_EVERY = 50
#: More than this fraction of the points failing ends the survey: in integers,
#: ``skipped * 10 > total``.
_SKIP_DENOMINATOR = 10
#: No stage travels a metre: a centre this far out is a unit slip (FM-32).
_MAX_CENTRE_UM = 1_000_000.0
#: The WARNING logged when the survey switches the autofocus off.
_AUTOFOCUS_LINE = (
    "the autofocus was switched off for the survey; engage it again before the scan"
)
_AUTOFOCUS_OFF_LINE = (
    "the autofocus is off after the survey; the scan never switches it on: lock it "
    "on the sample before the scan"
)
_NO_LIMITS_LINE = (
    "the stage reports no soft limits, so the survey's moves were not checked "
    "against any travel"
)


class PlateFindingParams(PluginParams):
    """The survey's extent and density, and which plates may be fitted."""

    plate: str = Field(
        default="96-well",
        description=(
            "The plate the survey is sized for: its extent, a margin of "
            "pitches past the outer wells. A useq plate name ending in '-well'."
        ),
    )
    plates: tuple[str, ...] = Field(
        default=(),
        description=(
            "The plates the survey is fitted against, JSON: "
            'plates=["96-well", "384-well"]. Empty: every registered well plate.'
        ),
    )
    centre_x_um: float | None = Field(
        default=None,
        ge=-_MAX_CENTRE_UM,
        le=_MAX_CENTRE_UM,
        description=(
            "Stage X of the plate's centre in µm, given with centre_y_um; "
            "default the stage's position."
        ),
    )
    centre_y_um: float | None = Field(
        default=None,
        ge=-_MAX_CENTRE_UM,
        le=_MAX_CENTRE_UM,
        description=(
            "Stage Y of the plate's centre in µm, given with centre_x_um; "
            "default the stage's position."
        ),
    )
    step_um: float | None = Field(
        default=None,
        gt=0.0,
        le=20_000.0,
        description=(
            "Distance between survey points in µm; default a third of the "
            "plate's pitch. At most 25 000 points are made."
        ),
    )
    margin_pitches: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description=(
            "How far past the outer wells the survey reaches on each side, in "
            "pitches. A survey that spans less than the outer wells plus one "
            "pitch is refused, because A1 cannot be pinned down: about 0.5 here, "
            "a little less with jitter."
        ),
    )
    jitter: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Random nudge of each point as a fraction of the step; a regular "
            "raster beats against the plate's lattice."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Seeds the nudge: the same seed gives the same survey.",
    )
    statistic: Literal["mean", "std"] = Field(
        default="mean",
        description=(
            "The number taken from each frame: its mean, or its standard "
            "deviation (for a contrast that sets wells apart by texture)."
        ),
    )
    max_failed_in_row: int = Field(
        default=3,
        ge=1,
        le=100,
        description=(
            "This many survey points failing in a row end the survey; the "
            "checkpoint keeps what was sampled."
        ),
    )

    @field_validator("plate")
    @classmethod
    def _plate_is_a_well_plate(cls, value: str) -> str:
        return _well_plate(value)

    @field_validator("plates")
    @classmethod
    def _plates_are_well_plates_once_each(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        for name in value:
            _well_plate(name)
        for name in dict.fromkeys(value):
            if value.count(name) > 1:
                raise ValueError(
                    f"plate-finding fits each plate once: {name} is given twice"
                )
        return value

    @model_validator(mode="after")
    def _the_centre_is_both_coordinates_or_neither(self) -> PlateFindingParams:
        if (self.centre_x_um is None) != (self.centre_y_um is None):
            raise ValueError(
                "give centre_x_um and centre_y_um together, or neither (the "
                "stage's position)"
            )
        return self


def _well_plate(name: str) -> str:
    """``name`` when it is a registered plate of wells; the others have no lattice to fit."""
    keys = sorted(
        str(k) for k in useq.registered_well_plate_keys() if str(k).endswith("-well")
    )
    if name not in keys:
        raise ValueError(
            f"plate-finding fits well plates: {name} is not one of {', '.join(keys)}"
        )
    return name


class SurveyCheckpoint(BaseModel):
    """What a resume needs: the centre, the survey's size, how far it got, and what it sampled.

    ``next_index`` counts the planned points handled, sampled or skipped, in the
    list left after the soft limits. That list is rebuilt from the limits each
    attempt reads, so ``total``, its size, is kept: a checkpoint that indexes a
    list rebuilt from state outside itself keeps the list's size, or other
    limits shift every index and nothing says so (FM-141). The count is not a
    fingerprint: limits that move and keep the count pass it. The centre is
    kept because without parameters it is the stage's position at the start, and
    on a resume the stage is elsewhere. A pydantic model built from Python objects does not
    validate a nested dataclass again, so a ``SurveyPoint`` holding ``nan`` is
    accepted when built and refused when the JSON is read back: the loop never
    builds one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    centre_um: tuple[float, float]
    total: int = Field(ge=0)
    next_index: int = Field(ge=0)
    skipped: int = Field(ge=0)
    points: tuple[SurveyPoint, ...]


class PlateFindingPlugin:
    """Surveys the plate in snake order and fits the plate type and A1 from the brightness."""

    name = _NAME
    summary = (
        "Find the plate automatically: survey it with the stage and the camera, "
        "and fit which plate it is and where A1 is."
    )
    Params = PlateFindingParams
    Result = PlateResult
    requires: tuple[type, ...] = (XYStage, Camera)
    provides = "plate"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: PlateFindingParams,
        ctx: RunContext,
    ) -> PlateResult:
        """Check, survey, fit; raise when what was found cannot be trusted.

        Raises:
            PluginRunError: A dry run was asked to survey; the survey is too big,
                or cannot pin A1 down inside the stage's soft limits (nothing
                moved); the checkpoint is not this survey's, or was made for
                another survey size; ``max_failed_in_row`` points in a row, or
                more than a tenth of them, failed; the fit is unusable, or its
                winner is a larger plate than the survey reaches.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped; the
                stage is where it is and the checkpoint keeps the last 50.
            StopRequested: The operator stopped it; ``--resume`` continues.
        """
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        if ctx.dry_run:
            raise PluginRunError(
                _NAME,
                "a dry run cannot survey the plate; run it without --dry-run",
            )

        checkpoint = ctx.checkpoint(SurveyCheckpoint)
        if checkpoint is not None:
            centre = checkpoint.centre_um
        elif params.centre_x_um is not None and params.centre_y_um is not None:
            centre = (params.centre_x_um, params.centre_y_um)
        else:
            # A stage still travelling from an earlier move reads a fraction of it.
            xy.wait()
            here = xy.position_um()
            centre = (here.x_um, here.y_um)

        planned = _plan(params, centre)
        kept, cut_clause, assumed = _inside_the_limits(xy, planned)
        total = len(kept)
        problem = survey_extent_note(kept, params.plate)
        if problem is None and total < 2:
            problem = (
                f"only {total} of {len(planned)} survey points are inside the "
                "stage's soft limits"
            )
        if problem is not None:
            raise PluginRunError(_NAME, problem + cut_clause)

        points, skipped, first = _resume(checkpoint, total)
        if checkpoint is None:
            # Before the first move: a survey stopped before its 50th point would
            # leave nothing to keep the centre, and a resume would survey again
            # around wherever the stage stopped (FM-111).
            _save(ctx, centre, total, 0, 0, [])
        autofocus = microscope.get(Autofocus)
        if first < total and autofocus is not None and autofocus.is_engaged():
            autofocus.disengage()
            ctx.logger.warning(_AUTOFOCUS_LINE)

        failed_in_row = 0
        for i in range(first, total):
            ctx.raise_if_stopped()
            x_um, y_um = kept[i]
            error: str | None = None
            value = math.nan
            arrived = (math.nan, math.nan)
            try:
                reached = xy.move_to_um(x_um, y_um)
                arrived = (reached.x_um, reached.y_um)
                value = frame_value(np.asarray(camera.snap()), params.statistic)
            except (MicroscopeHaltedError, MotionStoppedError):
                raise
            except HardwareError as exc:
                error = _one_line(exc)
            if error is None and not math.isfinite(value):
                error = f"the frame's {params.statistic} is not finite"
            if error is None and not all(math.isfinite(c) for c in arrived):
                error = "the stage's readback is not a finite position"
            if error is None:
                failed_in_row = 0
                # Where the stage says it was, not where it was sent: a stage
                # that stops short without an error would otherwise give A1 a
                # whole well off, and the extent checks would pass on the plan.
                points.append(SurveyPoint(arrived[0], arrived[1], value))
            else:
                skipped += 1
                failed_in_row += 1
                # In a row first: a stage that fails every move would otherwise
                # be left to fail a tenth of the survey, 113 moves of 1120 at a
                # 60 s timeout each.
                if failed_in_row >= params.max_failed_in_row:
                    head = f"{failed_in_row} survey points in a row"
                elif skipped * _SKIP_DENOMINATOR > total:
                    head = f"{skipped} of {total} survey points"
                else:
                    head = ""
                if head:
                    _save(ctx, centre, total, i + 1, skipped, points)
                    raise PluginRunError(
                        _NAME,
                        f"{head} could not be sampled, the last at "
                        f"({x_um + 0.0:.1f}, {y_um + 0.0:.1f}) µm: {error}; check "
                        "the stage, then run plate-finding --resume",
                    )
            ctx.progress(i + 1, total, "survey")
            if (i + 1) % _CHECKPOINT_EVERY == 0:
                _save(ctx, centre, total, i + 1, skipped, points)

        _save(ctx, centre, total, total, skipped, points)
        ctx.progress(total, total, "fitting")
        fits = fit(points, list(params.plates) or None)
        best, notes = decide(fits)
        if best is None:
            raise PluginRunError(_NAME, "; ".join(notes))
        # A winner larger than the survey, or a survey with its edge skipped,
        # leaves A1 unpinned however well the score reads.
        late = survey_extent_note([(p.x_um, p.y_um) for p in points], best.plate)
        if late is not None:
            raise PluginRunError(
                _NAME,
                f"{best.plate} fits best, but {late}: run plate-finding with "
                f"plate={best.plate}",
            )
        runner_up = fits[1] if len(fits) > 1 else None
        if skipped:
            assumed.append(
                f"{skipped} of {total} survey points could not be sampled and "
                "were skipped"
            )
        # After every check that can refuse the fit: a refused fit reads nothing.
        assumed.extend(_autofocus_lines(autofocus))
        ctx.logger.info(best.describe())
        return best.to_result(
            runner_up=runner_up.plate if runner_up is not None else None,
            margin=best.score - runner_up.score if runner_up is not None else None,
            notes=notes,
            assumed=assumed,
        )


def _plan(
    params: PlateFindingParams, centre: tuple[float, float]
) -> list[tuple[float, float]]:
    """The survey's points, or the refusal when it would be too big to make."""
    problem = ""
    try:
        return survey_points(
            params.plate,
            centre,
            step_um=params.step_um,
            margin_pitches=params.margin_pitches,
            jitter=params.jitter,
            seed=params.seed,
        )
    except ValueError as exc:
        problem = str(exc)
    # Raised outside the handler: nothing here keeps the arithmetic's error.
    raise PluginRunError(_NAME, problem)


def _inside_the_limits(
    xy: XYStage, planned: list[tuple[float, float]]
) -> tuple[list[tuple[float, float]], str, list[str]]:
    """The points the stage may visit, the clause that names its limits, and what is assumed.

    The test is the layer's own (``low <= v <= high`` on both axes), so a point
    kept here is one the move will not refuse. ``limits_um()`` is ``None`` when
    the stand does not say, and then nothing is cut and the result says that no
    travel was checked: what is true of the layer, not what to do about it. The
    clause is empty when the limits removed no point.
    """
    limits = xy.limits_um()
    if limits is None:
        return planned, "", [_NO_LIMITS_LINE]
    x_limits, y_limits = limits
    x_low, x_high = x_limits.low_um, x_limits.high_um
    y_low, y_high = y_limits.low_um, y_limits.high_um
    kept = [(x, y) for x, y in planned if x_low <= x <= x_high and y_low <= y <= y_high]
    cut = len(planned) - len(kept)
    if cut == 0:
        return kept, "", []
    clause = (
        f"; the stage's soft limits x [{x_low:g}, {x_high:g}], y [{y_low:g}, "
        f"{y_high:g}] µm cut the survey: put the plate nearer the middle of the "
        "travel, or change [safety] if the limits are wrong"
    )
    line = (
        f"{cut} of {len(planned)} survey points are outside the stage's soft "
        "limits and were not sampled"
    )
    return kept, clause, [line]


def _resume(
    checkpoint: SurveyCheckpoint | None, total: int
) -> tuple[list[SurveyPoint], int, int]:
    """The points, the skipped count and the index to continue from; a fresh start has none.

    A checkpoint is refused when it was made for a survey of another size (the
    soft limits changed, and every index with them), when it says more points
    were handled than the survey has, or when the points it holds and the ones
    it skipped do not add up to the points it says were handled: it is of
    another survey, or edited. The size is checked first: it is the reason the
    others would be wrong.
    """
    if checkpoint is None:
        return [], 0, 0
    if checkpoint.total != total:
        raise PluginRunError(
            _NAME,
            f"the checkpoint was made for a survey of {checkpoint.total} points "
            f"inside the soft limits and this one has {total}: the stage's soft "
            "limits changed; run plate-finding without --resume",
        )
    if checkpoint.next_index > total:
        raise PluginRunError(
            _NAME,
            f"the checkpoint has handled {checkpoint.next_index} points and this "
            f"survey has {total}: run plate-finding without --resume",
        )
    if checkpoint.skipped + len(checkpoint.points) != checkpoint.next_index:
        raise PluginRunError(
            _NAME,
            f"the checkpoint says {checkpoint.next_index} points were handled but "
            f"holds {len(checkpoint.points)} samples and {checkpoint.skipped} "
            "skipped: run plate-finding without --resume",
        )
    return list(checkpoint.points), checkpoint.skipped, checkpoint.next_index


def _save(
    ctx: RunContext,
    centre: tuple[float, float],
    total: int,
    next_index: int,
    skipped: int,
    points: list[SurveyPoint],
) -> None:
    ctx.save_checkpoint(
        SurveyCheckpoint(
            centre_um=centre,
            total=total,
            next_index=next_index,
            skipped=skipped,
            points=tuple(points),
        )
    )


def _autofocus_lines(autofocus: Autofocus | None) -> list[str]:
    """What the scan will find of the autofocus: its state now, read, never remembered.

    A flag kept from the switch-off would be silent when the operator switched
    it off, and stale when the operator locked it again. The price is a line on
    every survey of a stand whose autofocus is off, including when it was off
    before: the scan finds it off whoever did it. Only a read is made here: no
    ``engage``, no ``disengage`` (FM-88). A read that fails is a finding, not a
    lost survey, and a halt or a motion stop is not (FM-64).
    """
    if autofocus is None:
        return []
    try:
        engaged = autofocus.is_engaged()
    except (MicroscopeHaltedError, MotionStoppedError):
        raise
    except HardwareError as exc:
        problem = _one_line(exc)
    else:
        return [] if engaged else [_AUTOFOCUS_OFF_LINE]
    # Built outside the handler (FM-98).
    return [
        f"the autofocus's state could not be read after the survey ({problem}); "
        "check it before the scan"
    ]


def _one_line(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``: a skipped point's error, for a message."""
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}".rstrip()
