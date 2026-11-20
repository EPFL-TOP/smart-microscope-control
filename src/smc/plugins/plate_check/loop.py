"""The hardware loop that checks a registered plate against its registration (m3 §27.4).

``plate-check`` runs it alone, and the scan runs it before the first field of
every attempt (§27.5): one loop, so that what a check measures and refuses
cannot differ between the two. It is ``plate-walls``' own loop (§20.2) on two
wells the registration measured, at the objective in place, and a comparison of
their centres with the registered ones moved into that objective's frame (§18).

The decisions that keep the check from lying:

* **Everything is checked before anything moves** (FM-30): ``prepare_check``
  moves nothing and refuses a plate nobody measured the walls of, a well the
  registration did not measure, a frame or walk ``plate-walls`` would refuse,
  and a walk that leaves the soft limits.
* **Measured against measured.** The reference is the registration's measured
  centres, never the plate fit's prediction, which carried a residual of
  90.7 µm at session 5 (F8, §27.1).
* **The offset between objectives is the caller's to know, never guessed**:
  ``shift_um`` comes in measured, and a caller with none does not call (§27.1:
  39.4 µm on the demo for a plate that did not move).
* **Readbacks, not targets** (FM-142): ``measure_well`` places every frame
  where the stage says it was.
* **Nothing moves to put something back, and nothing touches the autofocus,
  the turret or Z** (FM-88, FM-223): the caller switched the autofocus off
  first and engages it again its own way. On any exception a WARNING says where
  the stage was left, and the exception goes on (FM-64).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import useq

from smc.hardware.capabilities import Camera, XYStage
from smc.plugins.api import RunContext
from smc.plugins.errors import PluginRunError
from smc.plugins.geometry import Matrix2
from smc.plugins.plate_check.algorithm import (
    DEFAULT_SEARCH_UM,
    farthest_pair,
    rigid_motion,
)
from smc.plugins.plate_walls.algorithm import STEP_FRACTION, walk_offsets_um
from smc.plugins.plate_walls.loop import (
    WallWalk,
    Where,
    check_frame,
    check_soft_limits,
    check_walk_size,
    checked_search,
    measure_well,
    warn_left,
)
from smc.plugins.slots import (
    CheckedWell,
    PlateCheck,
    PlateResult,
    WellFit,
    WellRef,
    normalise_well,
)

__all__ = [
    "CheckSetup",
    "PlateMovedError",
    "checked_line",
    "objective_text",
    "prepare_check",
    "run_check",
    "unsaved_line",
]

_MIN_WELLS = 2
_NO_TURRET = "no turret"


class PlateMovedError(PluginRunError):
    """The verdict of a check that measured: a registered well lies farther from its centre than the bound.

    A subclass of ``PluginRunError`` (its constructor and ``__reduce__``, so
    that the runner and every ``except PluginRunError`` see no change) so that
    the scan can tell the verdict from the check's other refusals: a plate that
    moved is answered by registering it again, and a check that could not
    measure (``measure_well``'s refusals: no wall in view, a wall lost, walls
    that fit no circle) is answered by scanning without it (m3 §27.4). Advice
    written for the first is never given to the second. A device's fault (a
    camera that stops answering) is neither: it keeps its own type (FM-64).
    """


@dataclass(frozen=True, slots=True)
class CheckSetup:
    """What one check measures and holds it to, fixed before the first move.

    ``refs`` are the expected centres (the registered ones plus ``shift_um``) in
    the order measured, ``registered_xy_um`` the registered ones in the same
    order. ``again`` ends the refusal: what the operator does about a plate that
    moved, which differs between the scan and the tool. ``assumed`` is the soft
    limits' line, or nothing.
    """

    plate: str
    plate_objective: str | None
    objective: str | None
    shift_um: tuple[float, float]
    refs: tuple[WellRef, ...]
    registered_xy_um: tuple[tuple[float, float], ...]
    walk: WallWalk
    offsets: tuple[float, ...]
    max_shift_um: float
    again: str
    assumed: tuple[str, ...]


def objective_text(objective: str | None) -> str:
    """The objective as the texts name it: ``no turret`` when there is none."""
    return objective if objective is not None else _NO_TURRET


def prepare_check(
    stage: str,
    xy: XYStage,
    plate: PlateResult,
    wells: tuple[str, ...],
    *,
    objective: str | None,
    shift_um: tuple[float, float],
    matrix: Matrix2,
    pixel_size_um: float,
    shape: tuple[int, int],
    max_shift_um: float,
    search_um: float | None,
    settle_s: float,
    folder: Path | None,
    folder_text: str | None,
    again: str,
) -> CheckSetup:
    """Everything the check needs, or the refusal; nothing moves.

    ``wells`` empty is the registration's pair farthest apart (§27.2 D1).

    Raises:
        PluginRunError: Of ``stage``, in this order: a plate whose walls were
            not measured (not ``plate-walls``), or whose record names two
            objectives; a well the registration did not measure, one given
            twice, fewer than two; a frame too small or that holds a whole
            well; a search past the radius; a walk too long; a walk that leaves
            the soft limits.
    """
    walls = plate.walls
    if plate.method != "wall-images" or walls is None:
        raise PluginRunError(
            stage,
            f"the plate was registered with {plate.method}: the check compares the "
            "centres plate-walls measured, so register it with plate-walls",
        )
    if walls.objective != plate.objective:
        # A record edited or mixed by hand: the shift is from the plate's
        # objective, and the centres are the wall images' (objective-offsets' rule).
        raise PluginRunError(
            stage,
            f"the plate's record names {objective_text(plate.objective)} and its wall "
            f"images {objective_text(walls.objective)}: register it again with "
            "plate-walls",
        )
    measured = {fit.well: fit.center_xy_um for fit in walls.wells}
    chosen = _chosen_wells(stage, wells, walls.wells, measured)

    diameter_um = float(useq.WellPlate.from_str(plate.plate).well_size[0]) * 1000.0
    check_frame(stage, matrix, shape, diameter_um)
    radius_um = diameter_um / 2.0
    search = checked_search(
        stage, search_um, radius_um, default_um=min(DEFAULT_SEARCH_UM, radius_um)
    )
    step_um = STEP_FRACTION * min(shape) * pixel_size_um
    check_walk_size(stage, len(chosen), search, step_um)
    offsets = walk_offsets_um(radius_um, search, step_um)

    registered = tuple(measured[name] for name in chosen)
    refs = tuple(
        WellRef(well=name, x_um=x + shift_um[0], y_um=y + shift_um[1])
        for name, (x, y) in zip(chosen, registered, strict=True)
    )
    assumed = check_soft_limits(stage, xy, refs, offsets)
    walk = WallWalk(
        stage=stage,
        plate=plate.plate,
        radius_um=radius_um,
        search_um=search,
        step_um=step_um,
        matrix=matrix,
        shape=shape,
        pixel_size_um=pixel_size_um,
        settle_s=settle_s,
        folder=folder,
        folder_text=folder_text,
        no_wall_hint=(
            "check that the light is on, and that the plate lies as it was "
            f"registered: a plate moved by more than {search:g} µm since "
            "plate-walls registered it hides {well}'s wall from this check; "
            f"{again}"
        ),
    )
    return CheckSetup(
        plate=plate.plate,
        plate_objective=plate.objective,
        objective=objective,
        shift_um=shift_um,
        refs=refs,
        registered_xy_um=registered,
        walk=walk,
        offsets=offsets,
        max_shift_um=max_shift_um,
        again=again,
        assumed=tuple(assumed),
    )


def _chosen_wells(
    stage: str,
    given: tuple[str, ...],
    fits: tuple[WellFit, ...],
    measured: dict[str, tuple[float, float]],
) -> tuple[str, ...]:
    """The wells to measure: those given, each measured by the registration and once; or the farthest pair."""
    names = [fit.well for fit in fits]
    if not given:
        if len(fits) < _MIN_WELLS:
            raise PluginRunError(
                stage,
                f"the registration measured {len(fits)} well, and the check needs "
                "two: register the plate again with plate-walls",
            )
        return farthest_pair([(fit.well, fit.center_xy_um) for fit in fits])
    chosen: list[str] = []
    for raw in given:
        name = normalise_well(raw)
        if name not in measured:
            raise PluginRunError(
                stage,
                f"{name} was not measured by the registration: give wells among "
                f"{', '.join(names)}",
            )
        if name in chosen:
            raise PluginRunError(stage, f"{name} is given twice in wells")
        chosen.append(name)
    if len(chosen) < _MIN_WELLS:
        raise PluginRunError(
            stage,
            "give at least two wells: one well shows a slide, and not a turn about itself",
        )
    return tuple(chosen)


def run_check(
    setup: CheckSetup,
    xy: XYStage,
    camera: Camera,
    ctx: RunContext,
    on_side: Callable[[str], None],
) -> PlateCheck:
    """Measure each well again, then hold the farthest one's distance to the bound.

    ``on_side("<well> <side>")`` is called after each side, for the caller's
    progress. The bound is judged after the last well, so that the refusal
    names every well's numbers (§27.4).

    Raises:
        PlateMovedError: Of ``setup.walk.stage``: a well moved more than
            ``max_shift_um`` (the refusal carries every well's distance and
            difference, the bound, the turn and the slide, then ``again``).
        PluginRunError: Of ``setup.walk.stage``, not a ``PlateMovedError``: any
            of ``measure_well``'s refusals (the check could not measure). The
            stage is where it was left.
        MicroscopeHaltedError, MotionStoppedError, StopRequested: Passed (FM-64).
    """
    stage = setup.walk.stage
    # A stage still travelling from an earlier move reads a fraction of it.
    xy.wait()
    start = xy.position_um()
    started = time.monotonic()
    where = Where()
    save_errors: list[str] = []
    checked: list[CheckedWell] = []
    frames = 0
    try:
        for ref, registered in zip(setup.refs, setup.registered_xy_um, strict=True):
            fit, taken = measure_well(
                xy,
                camera,
                ctx,
                setup.walk,
                ref,
                setup.offsets,
                where,
                save_errors,
                on_side,
            )
            frames += taken
            dx = fit.center_xy_um[0] - ref.x_um
            dy = fit.center_xy_um[1] - ref.y_um
            checked.append(
                CheckedWell(
                    well=ref.well,
                    registered_xy_um=registered,
                    expected_xy_um=(ref.x_um, ref.y_um),
                    fit=fit,
                    difference_xy_um=(dx, dy),
                    distance_um=math.hypot(dx, dy),
                )
            )
        checked_utc = datetime.now(timezone.utc)
        turn_deg, slide = rigid_motion(
            [w.expected_xy_um for w in checked],
            [w.fit.center_xy_um for w in checked],
        )
        worst_um = max(w.distance_um for w in checked)
        if worst_um > setup.max_shift_um:
            raise PlateMovedError(stage, _moved(setup, checked, turn_deg, slide))
    except BaseException:
        # A stop, a halt, a fault or a refusal: the stage stays where it is, the
        # exception goes on (FM-88, FM-64), and nothing is caught.
        warn_left(stage, ctx, xy, start, where)
        raise
    if save_errors:
        ctx.logger.warning(
            "%s: %d frames of the plate check could not be saved: %s",
            stage,
            len(save_errors),
            save_errors[0],
        )
    return PlateCheck(
        checked_utc=checked_utc,
        plate=setup.plate,
        plate_objective=setup.plate_objective,
        objective=setup.objective,
        shift_um=setup.shift_um,
        wells=tuple(checked),
        worst_um=worst_um,
        max_shift_um=setup.max_shift_um,
        turn_deg=turn_deg,
        slide_xy_um=slide,
        frames=frames,
        folder=setup.walk.folder_text,
        elapsed_s=time.monotonic() - started,
    )


def _moved(
    setup: CheckSetup,
    checked: list[CheckedWell],
    turn_deg: float,
    slide: tuple[float, float],
) -> str:
    """The refusal of a plate that moved: every well's distance and difference, the bound, the turn, the slide, then what to do.

    A distance is to the registered centre moved into the objective's frame
    (§18), so under another objective than the registration's the first well's
    clause names that offset: the numbers then do not match the record's, and
    the reader is told why (the text under the registration's own objective is
    unchanged).
    """
    parts = []
    for n, well in enumerate(checked):
        dx, dy = well.difference_xy_um
        lies = " lies" if n == 0 else ""
        whose = " from its registered centre" if n == 0 else ""
        if n == 0 and setup.shift_um != (0.0, 0.0):
            whose += (
                f" plus the offset ({setup.shift_um[0] + 0.0:+.1f}, "
                f"{setup.shift_um[1] + 0.0:+.1f}) µm to {objective_text(setup.objective)}"
            )
        parts.append(
            f"{well.well}{lies} {well.distance_um:.1f} µm{whose} "
            f"({dx + 0.0:+.1f}, {dy + 0.0:+.1f})"
        )
    return (
        f"the plate moved since plate-walls registered it: {_listed(parts)} under "
        f"{objective_text(setup.objective)}, more than {setup.max_shift_um:g} µm; it "
        f"turned about {turn_deg + 0.0:+.3f}° and slid about ({slide[0] + 0.0:+.1f}, "
        f"{slide[1] + 0.0:+.1f}) µm: {setup.again}"
    )


def _listed(parts: list[str]) -> str:
    """``a and b``, ``a, b and c``."""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def checked_line(check: PlateCheck, *, before_the_first_field: bool) -> str:
    """The ``assumed`` line of a check that passed: the scan's (§27.5) or the tool's (§27.6)."""
    wells = _listed([w.well for w in check.wells])
    objective = objective_text(check.objective)
    if before_the_first_field:
        return (
            f"the plate was checked before the first field: {wells} lie within "
            f"{check.worst_um:.1f} µm of their registered centres under {objective} "
            f"(bound {check.max_shift_um:g} µm), at "
            f"{check.checked_utc.isoformat(timespec='seconds')}"
        )
    return (
        f"the plate was checked under {objective}: {wells} within "
        f"{check.worst_um:.1f} µm of their registered centres (bound "
        f"{check.max_shift_um:g} µm)"
    )


def unsaved_line(check: PlateCheck) -> str | None:
    """The line for check frames that could not be saved, ``None`` when every one was or none was asked.

    A frame that could not be written has no file in a check whose frames are
    saved (``_save`` of ``plate-walls``' loop); the OS's reason is in the
    WARNING ``run_check`` logged, not in the result.
    """
    if check.folder is None:
        return None
    unsaved = sum(wall.file is None for well in check.wells for wall in well.fit.walls)
    if not unsaved:
        return None
    return f"{unsaved} frames of the plate check could not be saved (the log says why)"
