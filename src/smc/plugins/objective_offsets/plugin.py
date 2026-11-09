"""``objective-offsets`` measures the offsets between objectives from the plate itself (m3 §20).

The plate is registered with one objective (``plate-walls``, §17). This turns
the turret to each *target* objective, measures again under it the centres of
wells the registration measured, with ``plate-walls``' own loop and the target's
own camera calibration, and gives each target's offset from the registration's
objective: a well's centre under the target minus its centre under the
registration's objective (m2 §13.2). ``smc objective offset --from-run-doc``
shows what that puts into the profile's ``[objectives]``, and ``--write`` writes
it.

The decisions that keep a number that the stage will move by from lying:

* **Nothing turns before every check has passed, and a turn needs the
  operator's word** (FM-14): ``confirm_turns`` is required on every stand,
  whatever ``turret_requires_confirm`` says, because the tool turns several
  times over a loaded plate. The plate, the targets, the wells, the
  calibrations, the reference and the walks are all checked first (FM-30).
* **Measured, not predicted.** The offset is a well's *measured* centre under
  the target minus its *measured* centre under the registration's objective:
  the plate's fit is not used, since its prediction carries the fit's residual.
* **Each target has a calibration of its own.** The camera's turn and mirror
  decide which way every wall lies, and the scale changes with the objective.
* **A turn is read back, and Z must be still** (FM-167, FM-188, FM-195): the
  turret must read the target after it, and Z read twice 0.1 s apart must not
  move more than 0.5 µm. Nothing is imaged with an objective the stand did not
  settle on.
* **The wells must agree** (FM-73): their differences are the evidence that the
  shift is one translation over the plate. They are judged after every target
  is measured, so the run fails with all the numbers in one message and no
  ``done`` record exists to write a profile from.
* **Nothing moves or turns back** (FM-88): on any exception a WARNING says where
  the stage and the turret were left. No Z, autofocus or light is touched, and
  no Z offset is measured (a focused Z needs the operator, which a plugin cannot
  ask).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import useq
from pydantic import Field

from smc.hardware.capabilities import Camera, ObjectiveTurret, XYStage, ZStage
from smc.hardware.errors import SafetyRefusedError
from smc.hardware.profile import MAX_XY_OFFSET_UM
from smc.plugins.api import PluginParams, RunContext
from smc.plugins.camera_calibration.plugin import calibration_from
from smc.plugins.errors import PluginRunError
from smc.plugins.objective_offsets.algorithm import (
    MAX_TARGETS,
    SPREAD_PIXELS,
    default_max_spread_um,
    mean_shift_um,
    offsets_from_reference,
    spread_um,
)
from smc.plugins.plate_walls.algorithm import (
    SIDES,
    STEP_FRACTION,
    walk_offsets_um,
)
from smc.plugins.plate_walls.loop import (
    WallWalk,
    Where,
    check_calibration,
    check_frame,
    check_soft_limits,
    check_walk_size,
    checked_search,
    measure_well,
    move,
    warn_left,
)
from smc.plugins.slots import (
    CameraCalibration,
    MeasuredOffset,
    ObjectiveOffsetsResult,
    OffsetWell,
    PlateResult,
    TargetMeasure,
    WallImages,
    WellRef,
    normalise_well,
)
from smc.run.files import new_token

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["ObjectiveOffsetsParams", "ObjectiveOffsetsPlugin"]

_NAME = "objective-offsets"
_MAX_WELLS = 6
_MIN_WELLS = 2
#: After a turn, Z is read twice this many seconds apart and may not move more
#: than ``Z_STILL_UM``: m2 §8.11 D2's ``TURN_Z_SETTLE_S`` and ``TURN_Z_STILL_UM``,
#: copied with that reference because a plugin never imports a backend. Chosen,
#: not measured.
Z_STILL_S = 0.1
Z_STILL_UM = 0.5
_LEVER_LINE = (
    "the offsets are for the magnifier lever as it was during this run: the "
    "profile keys them by objective alone, for the lever at 1.0x (FM-103)"
)
_Z_LINE = (
    "Z after a turn is where the stand left it once it was still, not a focused "
    "Z: no z_offset_um is measured"
)
#: The tool compares each well's centre now with the one plate-walls recorded, so
#: a plate that moved as a whole in between (re-seated, bumped, drifted) shifts
#: every well alike: the wells agree and the shift reads as an offset between
#: objectives. Nothing in the run tells the two apart; the result says so.
_PLATE_LINE = (
    "the plate has not moved since plate-walls registered it: a plate that moved "
    "as a whole is taken for an offset between objectives, and the wells agreeing "
    "does not show otherwise"
)


class ObjectiveOffsetsParams(PluginParams):
    """The objectives to measure, one calibration of each, and the wells to measure them with."""

    targets: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_TARGETS,
        description=(
            "The objectives to measure, by turret label, in the order the turret "
            'turns to them (JSON: targets=["2-Plan Fluor 10x"]); the last one is '
            "left in place."
        ),
    )
    calibrations: tuple[Path, ...] = Field(
        min_length=1,
        max_length=MAX_TARGETS,
        description=(
            "One run document per target holding a camera-calibration done with "
            "that target in place (JSON list of paths, as typed; relative to the "
            "working directory, as from_run_doc)."
        ),
    )
    confirm_turns: bool = Field(
        False,
        description=(
            "The stage is clear for the turret to turn to each target with the "
            "plate on it (FM-14)."
        ),
    )
    wells: tuple[str, ...] = Field(
        (),
        max_length=_MAX_WELLS,
        description=(
            "Wells the registration measured, at least two; default all of them."
        ),
    )
    search_um: float | None = Field(
        None,
        gt=0.0,
        le=MAX_XY_OFFSET_UM,
        description=(
            "How far each wall may lie from where the registration's centre puts "
            "it, in µm; default min(1000, the well's radius); never above the "
            "radius. An offset larger than the profile's bound could not be "
            "written anyway."
        ),
    )
    max_spread_um: float | None = Field(
        None,
        gt=0.0,
        le=100.0,
        description=(
            "How far apart the wells' differences may lie, in µm; default "
            f"{SPREAD_PIXELS:g} pixels of the coarser objective."
        ),
    )
    settle_ms: float = Field(
        0.0,
        ge=0.0,
        le=10_000.0,
        description="Wait this long after each move before the frame is taken.",
    )
    save_frames: bool = Field(
        True,
        description=(
            "Save the frame centred on each wall as objective-offsets/<token>/"
            "<n>/<well>_<side>.tif beside the run document, n the target's place "
            "in targets."
        ),
    )


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What one target is measured with, fixed before the first turn."""

    target: str
    path: Path
    calibration: CameraCalibration
    measured: str
    radius_um: float
    search_um: float
    step_um: float
    offsets: tuple[float, ...]


class ObjectiveOffsetsPlugin:
    """Measures each target objective's offset from the plate's objective, on the wells the plate was registered with."""

    name = _NAME
    summary = (
        "Measure the offsets between objectives from the plate: re-find the "
        "registered wells under each target objective and give its shift."
    )
    Params = ObjectiveOffsetsParams
    Result = ObjectiveOffsetsResult
    requires: tuple[type, ...] = (XYStage, Camera, ObjectiveTurret)
    provides = "objective-offsets"
    reads: tuple[str, ...] = ("plate",)
    reads_optional: tuple[str, ...] = ()

    def run(
        self,
        microscope: Microscope,
        params: ObjectiveOffsetsParams,
        ctx: RunContext,
    ) -> ObjectiveOffsetsResult:
        """Check, turn to each target, re-measure the wells, and refuse what cannot be trusted.

        Raises:
            PluginRunError: A dry run, no ``confirm_turns``, a plate that
                ``plate-walls`` did not register, a target that is not on the
                turret, is the plate's objective or is given twice, a well the
                registration did not measure, a missing, doubled or unfit
                calibration, a profile reference this run does not measure, a
                walk that is too long or leaves the soft limits (nothing
                turned or moved); or, the turret and the stage then where they
                were left: a turn the layer refused, a turret that reads
                another label after it, a Z that is still moving, a wall that
                is not found, wells that do not agree on the offset, or an
                offset beyond the profile's bound.
            MicroscopeHaltedError, MotionStoppedError: The stand was stopped;
                the stage and the turret are where they are.
            StopRequested: The operator stopped it; they are where they are.
        """
        xy = microscope.require(XYStage)
        camera = microscope.require(Camera)
        turret = microscope.require(ObjectiveTurret)
        z_stage = microscope.get(ZStage)

        if ctx.dry_run:
            raise PluginRunError(
                _NAME,
                "a dry run cannot turn the turret or image the walls; run it "
                "without --dry-run",
            )
        if not params.confirm_turns:
            raise PluginRunError(
                _NAME,
                f"objective-offsets turns the turret to {', '.join(params.targets)} "
                "with the plate on the stage: check that each objective clears "
                "the plate (its working distance, the plate's height), then give "
                "confirm_turns=true",
            )

        plate = ctx.upstream("plate", PlateResult)
        origin_objective, walls = _checked_plate(plate)
        targets = _checked_targets(params.targets, origin_objective, turret)
        wells = _checked_wells(params.wells, walls)

        height, width = camera.image_shape()
        shape = (int(height), int(width))
        well_plate = useq.WellPlate.from_str(plate.plate)
        diameter_um = float(well_plate.well_size[0]) * 1000.0
        calibrations = _checked_calibrations(
            params.calibrations,
            targets,
            profile=microscope.profile.microscope.name,
            dry_run=ctx.dry_run,
            shape=shape,
            diameter_um=diameter_um,
        )

        reference = turret.offset(origin_objective).reference
        if (
            reference is not None
            and reference != origin_objective
            and reference not in targets
        ):
            raise PluginRunError(
                _NAME,
                f"the profile's offsets are measured from {reference} "
                "([objectives] reference), which this run does not measure: add "
                f"{reference} to targets, or clear the reference",
            )

        radius_um = diameter_um / 2.0
        refs = tuple(WellRef(well=w, x_um=c[0], y_um=c[1]) for w, c in wells)
        assumed: list[str] = []
        prepared: list[_Prepared] = []
        for target, path, calibration, measured in calibrations:
            search_um = checked_search(
                _NAME,
                params.search_um,
                radius_um,
                default_um=min(MAX_XY_OFFSET_UM, radius_um),
            )
            step_um = STEP_FRACTION * min(shape) * calibration.pixel_size_um
            check_walk_size(_NAME, len(refs), search_um, step_um)
            distances = walk_offsets_um(radius_um, search_um, step_um)
            line = check_soft_limits(_NAME, xy, refs, distances)
            if line and not assumed:
                assumed.extend(line)
            prepared.append(
                _Prepared(
                    target,
                    path,
                    calibration,
                    measured,
                    radius_um,
                    search_um,
                    step_um,
                    distances,
                )
            )

        # A stage still travelling from an earlier move reads a fraction of it.
        xy.wait()
        start = xy.position_um()
        started = time.monotonic()
        token = new_token()
        where = Where()
        save_errors: list[str] = []
        measures: list[TargetMeasure] = []
        sides_done = 0
        total_sides = len(prepared) * len(SIDES) * len(refs)
        try:
            for n, item in enumerate(prepared, start=1):
                ctx.raise_if_stopped()
                where.text = f"the turn to {item.target}"
                z_after_turn_um = _turn(turret, z_stage, item.target, ctx)

                walk = WallWalk(
                    stage=_NAME,
                    plate=plate.plate,
                    radius_um=item.radius_um,
                    search_um=item.search_um,
                    step_um=item.step_um,
                    matrix=item.calibration.matrix_um_per_px,
                    shape=shape,
                    pixel_size_um=item.calibration.pixel_size_um,
                    settle_s=params.settle_ms / 1000.0,
                    folder=(
                        ctx.run_dir / "objective-offsets" / token / str(n)
                        if params.save_frames
                        else None
                    ),
                    folder_text=(
                        f"objective-offsets/{token}/{n}" if params.save_frames else None
                    ),
                    no_wall_hint=(
                        "check that the light is on, that "
                        f"{item.target} images the wall as the registration's "
                        "objective did, and that search_um "
                        f"({item.search_um:g} µm) covers the offset"
                    ),
                )

                def on_side(text: str, target: str = item.target) -> None:
                    nonlocal sides_done
                    sides_done += 1
                    ctx.progress(sides_done, total_sides, f"{target}: {text}")

                offset_wells: list[OffsetWell] = []
                frames = 0
                for ref in refs:
                    fit, taken = measure_well(
                        xy,
                        camera,
                        ctx,
                        walk,
                        ref,
                        item.offsets,
                        where,
                        save_errors,
                        on_side,
                    )
                    origin = (ref.x_um, ref.y_um)
                    offset_wells.append(
                        OffsetWell(
                            well=ref.well,
                            origin_xy_um=origin,
                            delta_xy_um=(
                                fit.center_xy_um[0] - origin[0],
                                fit.center_xy_um[1] - origin[1],
                            ),
                            fit=fit,
                        )
                    )
                    frames += taken

                deltas = [w.delta_xy_um for w in offset_wells]
                bound_um = (
                    params.max_spread_um
                    if params.max_spread_um is not None
                    else default_max_spread_um(
                        walls.pixel_size_um, item.calibration.pixel_size_um
                    )
                )
                measures.append(
                    TargetMeasure(
                        objective=item.target,
                        calibration=item.path.as_posix(),
                        calibration_measured=item.measured,
                        pixel_size_um=item.calibration.pixel_size_um,
                        z_after_turn_um=z_after_turn_um,
                        wells=tuple(offset_wells),
                        shift_xy_um=mean_shift_um(deltas),
                        spread_um=spread_um(deltas),
                        max_spread_um=bound_um,
                        frames=frames,
                        folder=walk.folder_text,
                    )
                )

            _refuse_wells_that_disagree(measures)
            chosen, offsets = offsets_from_reference(
                origin_objective,
                [(m.objective, m.shift_xy_um) for m in measures],
                reference,
            )
            _refuse_offsets_beyond_the_profile(offsets)

            # The last target's last well, as that target measured it: a well centred
            # under the objective the turret was left at, which Live shows (§20.6).
            ctx.raise_if_stopped()
            last = measures[-1]
            last_fit = last.wells[-1].fit
            where.text = f"{last_fit.well}'s fitted centre under {last.objective}"
            move(_NAME, xy, *last_fit.center_xy_um, where.text)
        except BaseException:
            # A stop, a halt, a fault or a refusal: neither the stage nor the turret
            # is put back, and the exception goes on (FM-88, FM-64).
            warn_left(_NAME, ctx, xy, start, where)
            _warn_turret_left(ctx, turret)
            raise

        assumed.append(_LEVER_LINE)
        assumed.append(_Z_LINE)
        assumed.append(_PLATE_LINE)
        if reference is None:
            assumed.append(
                "the profile names no reference objective: the offsets are from "
                f"{origin_objective}, which smc objective offset --write makes "
                "the reference"
            )
        if save_errors:
            assumed.append(
                f"{len(save_errors)} frames could not be saved: {save_errors[0]}"
            )
        return ObjectiveOffsetsResult(
            plate_objective=origin_objective,
            reference=chosen,
            reference_in_profile=reference is not None,
            measures=tuple(measures),
            offsets=tuple(
                MeasuredOffset(objective=label, xy_offset_um=xy_um)
                for label, xy_um in offsets
            ),
            objective_in_place=prepared[-1].target,
            elapsed_s=time.monotonic() - started,
            assumed=tuple(assumed),
        )


def _checked_plate(plate: PlateResult) -> tuple[str, WallImages]:
    """The objective the plate was registered with and its wall images, or the refusal: only ``plate-walls`` measured centres.

    A plate registered by eye (``plate-registration``) or fitted
    (``plate-finding``) has positions nobody measured, so a difference of them
    would be a difference of guesses. The wall images' own objective must be the
    plate's: a record edited or mixed by hand names two.
    """
    if plate.method != "wall-images" or plate.walls is None:
        raise PluginRunError(
            _NAME,
            f"the plate was registered with {plate.method}: objective-offsets "
            "compares the centres plate-walls measured, so register it with "
            "plate-walls",
        )
    if not plate.objective:
        raise PluginRunError(
            _NAME,
            "the plate does not record the objective it was registered with: "
            "register it with plate-walls on a stand with a turret",
        )
    if plate.walls.objective != plate.objective:
        raise PluginRunError(
            _NAME,
            f"the plate's record names {plate.objective} and its wall images "
            f"{plate.walls.objective}: register it again with plate-walls",
        )
    return plate.objective, plate.walls


def _checked_targets(
    given: tuple[str, ...], origin_objective: str, turret: ObjectiveTurret
) -> tuple[str, ...]:
    """The targets, each on the turret, none the plate's own and none twice."""
    labels = [label for label in turret.labels() if label]
    seen: set[str] = set()
    for target in given:
        if target not in labels:
            raise PluginRunError(
                _NAME,
                f"objective {target} is not on the turret: {', '.join(labels)}",
            )
        if target == origin_objective:
            raise PluginRunError(
                _NAME,
                f"{target} is the objective the plate was registered with: its "
                "offset from itself is zero; leave it out of targets",
            )
        if target in seen:
            raise PluginRunError(_NAME, f"{target} is given twice in targets")
        seen.add(target)
    return given


def _checked_wells(
    given: tuple[str, ...], walls: WallImages
) -> tuple[tuple[str, tuple[float, float]], ...]:
    """The wells to measure with their centres under the registration's objective, at least two.

    Only wells ``plate-walls`` measured have a measured centre to compare with;
    the plate's fit is not used. Without ``wells``, every one it measured.
    """
    measured = {fit.well: fit.center_xy_um for fit in walls.wells}
    names = list(measured)
    if not given:
        chosen = names
    else:
        chosen = []
        for raw in given:
            name = normalise_well(raw)
            if name not in measured:
                raise PluginRunError(
                    _NAME,
                    f"{name} is not a well the registration measured: choose "
                    f"among {', '.join(names)}",
                )
            if name in chosen:
                raise PluginRunError(_NAME, f"{name} is given twice in wells")
            chosen.append(name)
    if len(chosen) < _MIN_WELLS:
        raise PluginRunError(
            _NAME,
            "give at least two wells: the offset is a translation only if they agree",
        )
    return tuple((name, measured[name]) for name in chosen)


def _checked_calibrations(
    paths: tuple[Path, ...],
    targets: tuple[str, ...],
    *,
    profile: str,
    dry_run: bool,
    shape: tuple[int, int],
    diameter_um: float,
) -> list[tuple[str, Path, CameraCalibration, str]]:
    """One usable calibration per target, in the order of ``targets``, or the refusal.

    Each comes from a run document of its own, by the rule ``camera-calibration``
    reuses one with. They are matched to the targets by the objective they
    record, never by their place in ``paths``: a calibration of another
    objective would invert every wall, and one in the wrong place is the likelier
    mistake.
    """
    found: dict[str, tuple[Path, CameraCalibration, str]] = {}
    for path in paths:
        calibration, measured = calibration_from(
            _NAME, path, profile=profile, dry_run=dry_run
        )
        objective = calibration.objective
        if objective is None or objective not in targets:
            raise PluginRunError(
                _NAME,
                f"the calibration in {path.as_posix()} is of "
                f"{objective if objective is not None else 'an unknown objective'}, "
                "which is not in targets: give one calibration per target",
            )
        if objective in found:
            raise PluginRunError(
                _NAME,
                f"{found[objective][0].as_posix()} and {path.as_posix()} are both "
                f"calibrations of {objective}: give one per target",
            )
        found[objective] = (path, calibration, measured)
    for target in targets:
        if target not in found:
            raise PluginRunError(
                _NAME,
                f"no camera calibration of {target} in calibrations: turn to it "
                f'(smc objective set "{target}" --confirm), run smc run '
                "camera-calibration --run-doc <a document of its own> on a "
                "well's wall, and add that document to calibrations",
            )
    ordered: list[tuple[str, Path, CameraCalibration, str]] = []
    for target in targets:
        path, calibration, measured = found[target]
        check_calibration(_NAME, calibration, target, True, shape)
        check_frame(_NAME, calibration.matrix_um_per_px, shape, diameter_um)
        ordered.append((target, path, calibration, measured))
    return ordered


def _turn(
    turret: ObjectiveTurret, z_stage: ZStage | None, target: str, ctx: RunContext
) -> float | None:
    """Turn to ``target``, check the turret says so and that Z is still; Z after the turn, ``None`` without a focus drive.

    The layer's own confirmation of the turn is given (``confirm=True``, which
    the tool has the operator's word for). Then the turret's readback must be
    the target (FM-167, FM-188: a turret whose label lags, or that did not turn,
    would give another objective's frames a target's name), and Z, which the
    Ti2-E moves on a turn by itself, must be still: two reads ``Z_STILL_S``
    apart may differ by ``Z_STILL_UM`` at most (m2 §8.11 D2). A Z read that
    raises is not caught: it is not "still".
    """
    problem = ""
    try:
        turret.select(target, confirm=True)
    except SafetyRefusedError as exc:
        problem = f"the turn to {target} was refused: {exc}"
    if problem:
        # Raised outside the handler: nothing here keeps the layer's error (FM-98).
        raise PluginRunError(_NAME, problem)

    now = turret.current()
    if now != target:
        raise PluginRunError(
            _NAME,
            f"the turret reads {now} after the turn to {target}: it did not turn, "
            "or its label lags (m2 §8.11 D3); nothing was measured with "
            f"{target}",
        )
    if z_stage is None:
        return None
    first = z_stage.position_um()
    ctx.wait_s(Z_STILL_S)
    second = z_stage.position_um()
    # A read that is not a number is refused first: a NaN compares false with
    # everything, so `abs(nan) > bound` would call it still and start the walk at
    # an unknown Z (FM-104, FM-195).
    if not (math.isfinite(first) and math.isfinite(second)):
        raise PluginRunError(
            _NAME,
            f"Z reads {first!r} and {second!r} µm after the turn to {target}: a Z "
            f"that is not a number is not a still Z; nothing was measured with "
            f"{target}",
        )
    if abs(second - first) > Z_STILL_UM:
        raise PluginRunError(
            _NAME,
            f"Z moved from {first:.2f} to {second:.2f} µm within {Z_STILL_S:g} s "
            f"after the turn to {target}: the stand was still moving it; nothing "
            f"was measured with {target}",
        )
    return second


def _refuse_wells_that_disagree(measures: list[TargetMeasure]) -> None:
    """One refusal for every target whose wells disagree beyond its bound, with all their numbers.

    Judged after the last target, so that the operator reads every number of the
    run at once, and so that the run fails: a run that did not finish has no
    ``done`` record for ``smc objective offset --write`` to write from.
    """
    failing = [m for m in measures if m.spread_um > m.max_spread_um]
    if not failing:
        return
    parts = []
    for measure in failing:
        wells = ", ".join(
            f"{w.well} ({w.delta_xy_um[0]:+.2f}, {w.delta_xy_um[1]:+.2f})"
            for w in measure.wells
        )
        parts.append(
            f"{measure.objective}: {wells} µm, spread {measure.spread_um:.2f} µm, "
            f"bound {measure.max_spread_um:.2f} µm"
        )
    folders = [m.folder for m in failing if m.folder is not None]
    look = (
        f"look at the frames in {', '.join(folders)}"
        if len(folders) == len(failing)
        else "the frames were not saved"
    )
    raise PluginRunError(
        _NAME,
        "the wells do not agree on the offset, so it is not a translation or not "
        f"measured well enough: {'; '.join(parts)}; {look}, or give a larger "
        "max_spread_um if the stage's repeatability is larger",
    )


def _refuse_offsets_beyond_the_profile(
    offsets: tuple[tuple[str, tuple[float, float]], ...],
) -> None:
    """Refuse an offset the profile could not hold: a wrong well or calibration, not two parcentric objectives."""
    for label, (x_um, y_um) in offsets:
        if max(abs(x_um), abs(y_um)) > MAX_XY_OFFSET_UM:
            raise PluginRunError(
                _NAME,
                f"the offset of {label} would be ({x_um:.1f}, {y_um:.1f}) µm, "
                f"beyond the profile's ±{MAX_XY_OFFSET_UM:g} µm: a wrong well or "
                "a wrong calibration, not two parcentric objectives",
            )


def _warn_turret_left(ctx: RunContext, turret: ObjectiveTurret) -> None:
    """Say which objective the turret was left at, reading it in a ``try`` of its own.

    A read that fails must not hide the error already on its way: the caller
    re-raises that one, whatever happens here.
    """
    try:
        label = turret.current()
    except Exception:
        label = "unknown"
    ctx.logger.warning("%s: the turret is left at %s, not turned back", _NAME, label)
