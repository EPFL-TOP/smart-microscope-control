"""Safety guards and the executor every hardware call goes through (design §7).

The guards live in the layer, not in the tools, so a tool cannot opt out of
them (CLAUDE.md, architecture rules):

* **Jog guard.** A relative XY move larger than ``max_jog_um`` is refused
  unless the call site passes ``force=True``. A mistyped relative move
  (``1000`` for ``100``) is the classic way to drive a stage out of the well
  or into the plate holder; an absolute move is not guarded, because
  crossing a plate is legitimate travel.
* **Soft limits.** A target outside the profile's travel range is refused
  and cannot be forced: the limits describe where the objective or the
  holder would collide, and no tool knows better at run time. No limits
  configured means no check, not a default range.
* **Non-finite targets** (``nan``, ``inf``) are refused whatever the limits:
  ``nan`` compares false against every bound and would otherwise slip
  through each check and reach the stage driver.

``Executor`` serialises hardware access behind one re-entrant lock and
implements dry-run. In dry-run, mutations are logged and skipped, but
*reads still reach the hardware*: a dry-run session shows the real stage
position and never moves it.

This module is pure: it imports no Micro-Manager code.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import TypeVar

from smc.hardware.errors import SafetyRefusedError

__all__ = ["Executor", "Safety"]

T = TypeVar("T")


def _check_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"{name} must be finite with low <= high, got {bounds!r}")
    return (float(low), float(high))


class Safety:
    """The guards for one microscope, built from plain numbers.

    Plain arguments keep this module independent of the profile model; the
    ``Microscope`` facade builds a ``Safety`` from ``profile.safety``.
    """

    def __init__(
        self,
        *,
        max_jog_um: float,
        z_soft_limits_um: tuple[float, float] | None = None,
        xy_soft_limits_um: tuple[tuple[float, float], tuple[float, float]]
        | None = None,
    ) -> None:
        if not math.isfinite(max_jog_um) or max_jog_um < 0:
            raise ValueError(f"max_jog_um must be finite and >= 0, got {max_jog_um!r}")
        self.max_jog_um = float(max_jog_um)
        self.z_soft_limits_um = (
            None
            if z_soft_limits_um is None
            else _check_range("z_soft_limits_um", z_soft_limits_um)
        )
        self.xy_soft_limits_um = (
            None
            if xy_soft_limits_um is None
            else (
                _check_range("xy_soft_limits_um[x]", xy_soft_limits_um[0]),
                _check_range("xy_soft_limits_um[y]", xy_soft_limits_um[1]),
            )
        )

    def check_jog_um(self, dx_um: float, dy_um: float, *, force: bool) -> None:
        """Refuse a relative XY move above ``max_jog_um`` unless forced.

        Raises:
            SafetyRefusedError: The jog is too large (forceable), or not finite
                (never forceable).
        """
        if not (math.isfinite(dx_um) and math.isfinite(dy_um)):
            raise SafetyRefusedError(
                f"jog ({dx_um}, {dy_um}) µm is not a finite distance"
            )
        size = max(abs(dx_um), abs(dy_um))
        if size > self.max_jog_um and not force:
            raise SafetyRefusedError(
                f"jog ({dx_um}, {dy_um}) µm exceeds the jog limit of {self.max_jog_um} µm",
                how_to_force="pass force=True",
            )

    def check_xy_target_um(self, x_um: float, y_um: float) -> None:
        """Refuse an XY target outside the soft limits; never forceable.

        Raises:
            SafetyRefusedError: The target is outside the limits or not finite.
        """
        if not (math.isfinite(x_um) and math.isfinite(y_um)):
            raise SafetyRefusedError(
                f"XY target ({x_um}, {y_um}) µm is not a finite position"
            )
        if self.xy_soft_limits_um is None:
            return
        (x_low, x_high), (y_low, y_high) = self.xy_soft_limits_um
        if not (x_low <= x_um <= x_high and y_low <= y_um <= y_high):
            raise SafetyRefusedError(
                f"XY target ({x_um}, {y_um}) µm is outside the soft limits "
                f"x [{x_low}, {x_high}], y [{y_low}, {y_high}] µm; change "
                f"[safety] in the profile if the limits are wrong"
            )

    def check_z_target_um(self, z_um: float) -> None:
        """Refuse a Z target outside the soft limits; never forceable.

        Raises:
            SafetyRefusedError: The target is outside the limits or not finite.
        """
        if not math.isfinite(z_um):
            raise SafetyRefusedError(f"Z target {z_um} µm is not a finite position")
        if self.z_soft_limits_um is None:
            return
        low, high = self.z_soft_limits_um
        if not low <= z_um <= high:
            raise SafetyRefusedError(
                f"Z target {z_um} µm is outside the soft limits [{low}, {high}] µm; "
                f"change [safety] in the profile if the limits are wrong"
            )


class Executor:
    """Runs every backend call under the microscope's lock, honouring dry-run.

    One ``RLock`` per microscope makes the capabilities safe to call from a
    UI thread and a worker at once (they stay sequential); re-entrant, so a
    mutation can read its own readback and wait for its device inside the
    same critical section.
    """

    def __init__(
        self, *, dry_run: bool, lock: threading.RLock, logger: logging.Logger
    ) -> None:
        self._dry_run = dry_run
        self._lock = lock
        self._logger = logger

    @property
    def dry_run(self) -> bool:
        """Whether mutations are skipped; fixed at construction."""
        return self._dry_run

    def do(self, description: str, action: Callable[[], T], *, dry_result: T) -> T:
        """Log and perform a mutation; in dry-run, log it and return ``dry_result``."""
        with self._lock:
            if self._dry_run:
                self._logger.info("[dry-run] %s", description)
                return dry_result
            self._logger.info("%s", description)
            return action()

    def read(self, action: Callable[[], T]) -> T:
        """Perform a read under the lock; reads reach the hardware even in dry-run."""
        with self._lock:
            return action()
