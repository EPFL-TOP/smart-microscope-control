"""The plate in another objective's frame (m3 §18; the offsets are m2 §13).

A plate registered with the 4x or the 10x is scanned with the 40x, and two
objectives do not centre the same feature at the same stage position. The
offsets between them are the profile's, handed out by
``ObjectiveTurret.offset()``; this module moves a registered plate from the
frame of the objective it was registered with to the frame of the objective
that uses it. Only XY moves: the layer applies no Z offset (m2 §13.2).

What it decides, because a stage will visit the positions it hands out:

* **Unknown moves unshifted and says so** (the rules of ``plate_for_objective``).
  An objective that was not recorded, an offset that was not measured and a
  stand with no turret each give the registration's own plate and a line for
  ``assumed``; none gives a shift of zero as if it had been measured.
* **The same objective is the same frame**, with no line and no question put
  to the profile: nothing is shifted that need not be.
* **A translated plate is validated, never copied.** It is built once by
  ``PlateResult.model_validate``, so a shift that would put A1 at ``nan`` is
  refused as any other plate is; ``model_copy`` would skip that.
* **An empty turret position is no objective** (FM-125): ``""`` is refused by
  ``objective_for``, in every stage that asks which objective is in use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from smc.plugins.errors import PluginRunError
from smc.plugins.slots import PlateResult

if TYPE_CHECKING:
    import logging

    from smc.hardware.capabilities import ObjectiveOffset, ObjectiveTurret

__all__ = [
    "PlateInFrame",
    "log_frame",
    "objective_for",
    "objective_shift_um",
    "plate_for_objective",
]


def objective_for(
    stage: str, requested: str | None, turret: ObjectiveTurret | None
) -> tuple[str | None, bool]:
    """The objective a stage works with, and whether it is the one in place.

    Without a turret the objective is unknown (``None``) and the camera's size is
    the only one there is, so it counts as in place. An objective that is asked
    for and cannot be checked (no turret) is refused: a plan for an objective
    that may not be there is the plan this stage exists not to make. So is a
    turret at an empty position, whose label is ``""`` (the protocol lists empty
    positions): that is no objective, and taking it for one would plan with the
    camera's size for an "objective" that does not exist. A turret that cannot be
    read raises its own error, which is not caught: it is not "no objective"
    (FM-19).

    Raises:
        PluginRunError: The turret is at an empty position, the stand has no
            turret and an objective was asked for, or the objective asked for is
            not on the turret.
    """
    if requested is None:
        if turret is None:
            return None, True
        current = turret.current()
        if not current:
            raise PluginRunError(
                stage,
                "no objective is in place: the turret is at an empty position; "
                "turn it to an objective, or give objective to plan for one",
            )
        return current, True
    if turret is None:
        raise PluginRunError(
            stage,
            "the stand has no turret, so objective cannot be checked; leave it unset",
        )
    known = [label for label in turret.labels() if label]
    if requested not in known:
        raise PluginRunError(
            stage, f"objective {requested!r} is not on the turret: {', '.join(known)}"
        )
    return requested, requested == turret.current()


def objective_shift_um(
    origin: ObjectiveOffset, target: ObjectiveOffset
) -> tuple[float, float] | None:
    """What to add to a position found with ``origin`` to have it with ``target``, in µm.

    ``target.xy_um - origin.xy_um`` per axis (m2 §13.2: ``p_O = p_reference +
    offset(O)``). ``None`` when either offset is not measured, or when the two
    were measured from different references, which say nothing about each other.
    """
    if origin.xy_um is None or target.xy_um is None:
        return None
    if origin.reference != target.reference:
        return None
    return (target.xy_um[0] - origin.xy_um[0], target.xy_um[1] - origin.xy_um[1])


@dataclass(frozen=True, slots=True)
class PlateInFrame:
    """A plate in the frame of the objective in use: the plate, the shift applied and what it assumed.

    ``plate`` is the registration's own when ``shift_um`` is ``(0.0, 0.0)`` (the
    same objective) or ``None`` (nothing could be applied), and a translated
    copy otherwise. ``assumed`` is empty when the frame is known, and one line
    when it is not: the caller adds it to its own ``assumed``.
    """

    plate: PlateResult
    #: The objective in use, as the caller gave it.
    objective: str | None
    shift_um: tuple[float, float] | None
    assumed: tuple[str, ...]


def plate_for_objective(
    plate: PlateResult, objective: str | None, turret: ObjectiveTurret | None
) -> PlateInFrame:
    """``plate`` as ``objective`` sees it, by the offsets the turret hands out.

    In this order:

    1. ``objective`` is the plate's own (both unknown included): the plate itself,
       no shift, no line.
    2. The plate does not record its objective: the plate itself, unshifted.
    3. The objective in use is unknown: the plate itself, unshifted.
    4. There is no turret, or the offset between the two is not measured: the plate
       itself, unshifted.
    5. Otherwise the plate translated: A1 and every reference move by the shift, the
       rotation and everything else stay, and the plate's objective becomes
       ``objective``.

    Rules 2 to 4 say so in ``assumed``: the positions are the registration's, not
    corrected. The caller logs that line as a WARNING (``log_frame``).

    Raises:
        pydantic.ValidationError: The translated plate is refused by validation
            (a shift that makes a position not finite).
    """
    if objective == plate.objective:
        return PlateInFrame(plate, objective, (0.0, 0.0), ())
    if plate.objective is None:
        return PlateInFrame(
            plate,
            objective,
            None,
            (
                "the plate does not record the objective it was registered with: no "
                f"offset between objectives is applied for {objective}",
            ),
        )
    if objective is None:
        return PlateInFrame(
            plate,
            None,
            None,
            (
                "the objective in use is unknown: the plate's positions are "
                f"{plate.objective}'s, and no offset between objectives is applied",
            ),
        )
    shift = (
        None
        if turret is None
        else objective_shift_um(
            turret.offset(plate.objective), turret.offset(objective)
        )
    )
    if shift is None:
        return PlateInFrame(
            plate,
            objective,
            None,
            (
                f"the offset between {plate.objective} and {objective} is not "
                "measured (the profile's [objectives]): the positions are "
                f"{plate.objective}'s, not corrected",
            ),
        )
    return PlateInFrame(_translated(plate, objective, shift), objective, shift, ())


def _translated(
    plate: PlateResult, objective: str, shift_um: tuple[float, float]
) -> PlateResult:
    """``plate`` moved by ``shift_um``: A1 and every reference; the rotation is a plate's, not an objective's.

    The shift is a translation, the same for every position on the stage (m2
    §13.2), so moving A1 moves every well centre and every wall by exactly the
    shift. It is built through ``model_validate``, which ``model_copy`` is not.
    """
    dx_um, dy_um = shift_um
    dump = plate.model_dump()
    dump["a1_center_xy_um"] = (
        plate.a1_center_xy_um[0] + dx_um,
        plate.a1_center_xy_um[1] + dy_um,
    )
    dump["refs"] = [
        {**ref, "x_um": ref["x_um"] + dx_um, "y_um": ref["y_um"] + dy_um}
        for ref in dump["refs"]
    ]
    dump["objective"] = objective
    return PlateResult.model_validate(dump)


def log_frame(
    logger: logging.Logger,
    plate: PlateResult,
    frame: PlateInFrame,
    turret: ObjectiveTurret | None,
) -> None:
    """Say what ``plate_for_objective`` did: its line as a WARNING, a measured shift as INFO.

    A line in ``assumed`` is read in the run document afterwards; the operator
    at the console reads the log, and an unshifted move is the one to see. The
    INFO line names the shift and when each offset was measured, so that a
    position that is off can be traced to the numbers it came from.
    """
    for line in frame.assumed:
        logger.warning(line)
    if frame.plate is plate or turret is None or frame.shift_um is None:
        return
    # Only a translated plate gets here, so both objectives are known.
    assert plate.objective is not None
    assert frame.objective is not None
    logger.info(
        "the plate registered with %s is moved into the frame of %s by (%+.2f, "
        "%+.2f) µm; offsets measured: %s %s, %s %s",
        plate.objective,
        frame.objective,
        frame.shift_um[0] + 0.0,
        frame.shift_um[1] + 0.0,
        plate.objective,
        turret.offset(plate.objective).measured or "(the reference)",
        frame.objective,
        turret.offset(frame.objective).measured or "(the reference)",
    )
