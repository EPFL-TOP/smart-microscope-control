"""The focus sweep, shared by ``focus-on-cells`` and ``scan`` (m3 §21.4).

The sweep moves Z with the autofocus **off** (D1): the 40x's offset factor is
not measured, so a step through the PFS offset is refused, and a Z move under
a lock is undone by the relock before the snap. The caller switches the
autofocus off with ``switch_off_for_sweep`` and checks the switch-off took:
an autofocus installed with ``Microscope.override`` (the scan's tests do) is
not reached by the layer's Z interlock (m2 §3.4).

Nothing here catches or restores anything. A halt, a stop and an operator's
stop pass as they are (FM-64), and so does every ``HardwareError``; after a
failure Z stays where it was read last and the autofocus stays off, because
putting either back would move Z after a stop (FM-88).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from smc.hardware.capabilities import Autofocus, Camera, Limits, ZStage
from smc.hardware.safety import DEFAULT_MAX_Z_JOG_UM
from smc.plugins.cell_focus.algorithm import (
    LANDING_FRACTION,
    ScoreName,
    first_outside,
    plan_planes,
    score_frame,
)
from smc.plugins.errors import PluginRunError

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = [
    "AutofocusStillOn",
    "FocusDriveOffTarget",
    "SweepBounds",
    "check_sweep_bounds",
    "move_checked",
    "outside_limits",
    "sweep",
    "switch_off_for_sweep",
]


@dataclass(frozen=True)
class SweepBounds:
    """What the checks before a sweep read: the soft limits and the Z the first sweep is centred on."""

    limits: Limits
    z_start_um: float


def outside_limits(
    centre_um: float, planes_um: Sequence[float], limits: Limits
) -> str | None:
    """The refusal of a sweep with a plane outside the soft limits (D3), or ``None``."""
    plane = first_outside(planes_um, limits.low_um, limits.high_um)
    if plane is None:
        return None
    return (
        f"the focus sweep around {centre_um:g} µm would reach {plane:g} µm, "
        f"outside the soft limits [{limits.low_um:g}, {limits.high_um:g}] µm"
    )


def check_sweep_bounds(
    stage: str,
    subject: str,
    microscope: Microscope,
    z: ZStage,
    range_name: str,
    range_um: float,
    step_um: float,
) -> SweepBounds:
    """§21.6's checks 2-4, before anything moves (dry run included): the limits, the reach, the planes.

    ``subject`` names what moves Z in the refusal (``cell_focus`` for the
    scan's parameter, ``focus-on-cells`` for the tool). The sweep's moves are
    absolute, so the layer's Z jog guard does not reach them (M1 §7): the
    plugin bounds their reach by the same number, the profile's
    ``max_z_jog_um``, else the layer's default.

    Raises:
        PluginRunError: The stand has no Z soft limits (unknown is not
            unlimited), ``range_um`` exceeds the Z jog limit, or a plane of the
            sweep around the Z read now lies outside the limits.
    """
    limits = z.limits_um()
    if limits is None:
        raise PluginRunError(
            stage,
            f"{subject} moves Z: the profile must set [safety] z_soft_limits_um",
        )
    configured = microscope.profile.safety.max_z_jog_um
    jog_um = DEFAULT_MAX_Z_JOG_UM if configured is None else float(configured)
    if range_um > jog_um:
        raise PluginRunError(
            stage,
            f"{range_name}={range_um:g} µm is above the stand's Z jog limit of "
            f"{jog_um:g} µm ([safety] max_z_jog_um): a sweep reaches no farther "
            "than a Z jog may",
        )
    z_start = z.position_um()
    if not math.isfinite(z_start):
        # FM-104: a drive that reads NaN has no plane to sweep around.
        raise PluginRunError(
            stage, f"Z reads {z_start!r}: {subject} cannot centre a sweep on it"
        )
    planes = plan_planes(z_start, range_um, step_um)
    refusal = outside_limits(z_start, planes, limits)
    if refusal is not None:
        raise PluginRunError(stage, refusal)
    return SweepBounds(limits, z_start)


class FocusDriveOffTarget(Exception):  # noqa: N818 (the name is fixed by m3 §21.4; it reads as a field's error type)
    """A Z readback farther from its command than the sweep allows."""


class AutofocusStillOn(Exception):  # noqa: N818 (see FocusDriveOffTarget)
    """The autofocus still reads on after the sweep switched it off."""


def switch_off_for_sweep(af: Autofocus) -> None:
    """Switch the autofocus off for a sweep, and read it back off.

    Never called in a dry run: ``disengage()`` is a safe call that acts in
    dry run too (m2 §3.2).

    Raises:
        AutofocusStillOn: It still reads engaged; nothing was moved.
    """
    af.disengage()
    if af.is_engaged():
        raise AutofocusStillOn(
            "the autofocus still reads on after it was switched off for the "
            "focus sweep: nothing was moved"
        )


def move_checked(z: ZStage, target_um: float, tolerance_um: float) -> float:
    """Move Z to ``target_um`` with the autofocus left off; the readback, checked (FM-167).

    A move that returned is not a move that arrived: the readback must lie
    within ``tolerance_um`` of the command, and a NaN readback is off target
    (FM-104).

    Raises:
        FocusDriveOffTarget: The readback is farther than ``tolerance_um``.
    """
    readback = z.move_to_um(target_um, keep_autofocus=False)
    if not abs(readback - target_um) <= tolerance_um:
        raise FocusDriveOffTarget(
            f"Z read {readback:.2f} µm after a move to {target_um:.2f} µm, more "
            f"than {tolerance_um:.2f} µm away"
        )
    return readback


def sweep(
    z: ZStage,
    camera: Camera,
    planes_um: Sequence[float],
    step_um: float,
    score: ScoreName,
    *,
    raise_if_stopped: Callable[[], None],
    on_plane: Callable[[int, float, npt.NDArray[Any]], None] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Move to each plane, snap and score; the readbacks and the scores, in the planes' order.

    ``on_plane(k, readback, frame)`` runs after each plane's score: the tool
    writes the frame there, so a failure at a later plane keeps the frames
    already taken. The autofocus must already be off (``switch_off_for_sweep``).

    Raises:
        FocusDriveOffTarget: A readback was farther than ``LANDING_FRACTION``
            of the step from its plane.
        HardwareError, MicroscopeHaltedError, MotionStoppedError, StopRequested:
            Passed as they are; nothing is restored.
    """
    tolerance_um = LANDING_FRACTION * step_um
    readbacks: list[float] = []
    scores: list[float] = []
    for k, plane in enumerate(planes_um):
        raise_if_stopped()
        readback = move_checked(z, plane, tolerance_um)
        frame = np.asarray(camera.snap())
        readbacks.append(readback)
        scores.append(score_frame(frame, score))
        if on_plane is not None:
            on_plane(k, readback, frame)
    return tuple(readbacks), tuple(scores)
