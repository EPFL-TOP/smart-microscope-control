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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from smc.hardware.capabilities import Autofocus, Camera, ZStage
from smc.plugins.cell_focus.algorithm import LANDING_FRACTION, ScoreName, score_frame

__all__ = [
    "AutofocusStillOn",
    "FocusDriveOffTarget",
    "move_checked",
    "sweep",
    "switch_off_for_sweep",
]


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
