"""Micro-Manager implementations of the capabilities (design §6).

Each class wraps one device label of a ``CMMCorePlus`` and nothing else: it
does not choose the device (roles do, #6) and does not own the core (the
facade does, #8). Every call goes through the microscope's ``Executor``, so
all of them share one lock and honour dry-run.

Hardware behaviour encoded here:

* **Explicit labels.** Stage and shutter calls pass the device label to
  MMCore instead of relying on the core's "current" XY/focus/shutter slot, so
  a stand with several stages (the Ti2 has four ``XYStage`` devices, FM-13)
  moves the one the role names.
* **Readback.** A mutation returns what the device reports afterwards, not
  what was asked: stages round to their step size, cameras clamp exposure.
* **Waiting.** ``wait`` polls ``deviceBusy`` every 10 ms against a deadline
  (default: the core timeout, which the facade raises to 60 s because a
  plate traverse outlasts MMCore's 5 s default, FM-10) and raises
  ``DeviceTimeoutError`` with what to change, never hangs forever.
* **Pixel size is measured or unknown.** MMCore's calibrated value when it
  has one, else the profile's value for the current objective, else ``0.0``.
* **Dry-run reads real state.** Safety checks and the position a relative
  move starts from use live reads; only the mutation is skipped.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from smc.hardware.capabilities import XY, Limits, PropertyInfo
from smc.hardware.errors import DeviceTimeoutError, HardwareError

if TYPE_CHECKING:
    import numpy as np
    from pymmcore_plus import CMMCorePlus

    from smc.hardware.safety import Executor, Safety

__all__ = ["MMCamera", "MMProperties", "MMShutter", "MMXYStage", "MMZStage"]

logger = logging.getLogger("smc.hardware.backends.mm")

#: How often ``wait`` asks the device whether it is still busy.
POLL_INTERVAL_S = 0.01


def _wait_until_idle(core: CMMCorePlus, label: str, timeout_s: float | None) -> None:
    """Poll ``deviceBusy`` until idle or the deadline; raise ``DeviceTimeoutError``."""
    if timeout_s is None:
        timeout_s = core.getTimeoutMs() / 1000.0
    if not math.isfinite(timeout_s) or timeout_s < 0:
        # nan would make the deadline comparison always false: an endless wait.
        raise ValueError(f"timeout_s must be finite and >= 0, got {timeout_s!r}")
    deadline = time.monotonic() + timeout_s
    while core.deviceBusy(label):
        if time.monotonic() >= deadline:
            raise DeviceTimeoutError(
                f"{label} is still busy after {timeout_s:g} s. If the move is "
                f"legitimately long, raise [micromanager] device_timeout_ms in "
                f"the profile; otherwise check the device and its cabling."
            )
        time.sleep(POLL_INTERVAL_S)


class MMXYStage:
    """``XYStage`` over one Micro-Manager XY stage device."""

    def __init__(
        self, core: CMMCorePlus, label: str, executor: Executor, safety: Safety
    ) -> None:
        self._core = core
        self._label = label
        self._executor = executor
        self._safety = safety

    def _read_position(self) -> XY:
        return XY(
            float(self._core.getXPosition(self._label)),
            float(self._core.getYPosition(self._label)),
        )

    def position_um(self) -> XY:
        """Where the stage reports it is."""
        return self._executor.read(self._read_position)

    def _move(self, description: str, x_um: float, y_um: float, wait: bool) -> XY:
        def action() -> XY:
            self._core.setXYPosition(self._label, x_um, y_um)
            if wait:
                _wait_until_idle(self._core, self._label, None)
            return self._read_position()

        return self._executor.do(description, action, dry_result=XY(x_um, y_um))

    def move_to_um(self, x_um: float, y_um: float, *, wait: bool = True) -> XY:
        """Move to an absolute position (soft limits apply, no jog guard)."""
        x_um, y_um = float(x_um), float(y_um)
        self._safety.check_xy_target_um(x_um, y_um)
        return self._move(f"xy_stage: move_to ({x_um}, {y_um}) µm", x_um, y_um, wait)

    def move_by_um(
        self, dx_um: float, dy_um: float, *, wait: bool = True, force: bool = False
    ) -> XY:
        """Move relative to the live position; jog-guarded unless ``force``.

        The target is computed from a live read and moved to absolutely, so the
        soft-limit check sees the real destination, and the whole
        read-check-move runs under the lock (another thread cannot move the
        stage in between).
        """
        dx_um, dy_um = float(dx_um), float(dy_um)
        self._safety.check_jog_um(dx_um, dy_um, force=force)

        def relative() -> XY:
            here = self._read_position()
            x_um, y_um = here.x_um + dx_um, here.y_um + dy_um
            self._safety.check_xy_target_um(x_um, y_um)
            return self._move(
                f"xy_stage: move_by ({dx_um}, {dy_um}) µm to ({x_um}, {y_um}) µm",
                x_um,
                y_um,
                wait,
            )

        return self._executor.read(relative)

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until idle; ``DeviceTimeoutError`` after ``timeout_s`` (default: core timeout)."""
        self._executor.read(
            lambda: _wait_until_idle(self._core, self._label, timeout_s)
        )

    def is_busy(self) -> bool:
        """Whether the stage reports it is still moving."""
        return self._executor.read(lambda: bool(self._core.deviceBusy(self._label)))

    def limits_um(self) -> tuple[Limits, Limits] | None:
        """The configured soft limits as ``(x, y)``; ``None`` when none are configured."""
        limits = self._safety.xy_soft_limits_um
        if limits is None:
            return None
        (x_low, x_high), (y_low, y_high) = limits
        return (Limits(x_low, x_high), Limits(y_low, y_high))


class MMZStage:
    """``ZStage`` over one Micro-Manager stage (focus) device."""

    def __init__(
        self, core: CMMCorePlus, label: str, executor: Executor, safety: Safety
    ) -> None:
        self._core = core
        self._label = label
        self._executor = executor
        self._safety = safety

    def _read_position(self) -> float:
        return float(self._core.getPosition(self._label))

    def position_um(self) -> float:
        """Where the drive reports it is."""
        return self._executor.read(self._read_position)

    def _move(self, description: str, z_um: float, wait: bool) -> float:
        def action() -> float:
            self._core.setPosition(self._label, z_um)
            if wait:
                _wait_until_idle(self._core, self._label, None)
            return self._read_position()

        return self._executor.do(description, action, dry_result=z_um)

    def move_to_um(self, z_um: float, *, wait: bool = True) -> float:
        """Move to an absolute position (soft limits apply)."""
        z_um = float(z_um)
        self._safety.check_z_target_um(z_um)
        return self._move(f"z: move_to {z_um} µm", z_um, wait)

    def move_by_um(self, dz_um: float, *, wait: bool = True) -> float:
        """Move relative to the live position (soft limits apply to the target)."""
        dz_um = float(dz_um)

        def relative() -> float:
            z_um = self._read_position() + dz_um
            self._safety.check_z_target_um(z_um)
            return self._move(f"z: move_by {dz_um} µm to {z_um} µm", z_um, wait)

        return self._executor.read(relative)

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until idle; ``DeviceTimeoutError`` after ``timeout_s`` (default: core timeout)."""
        self._executor.read(
            lambda: _wait_until_idle(self._core, self._label, timeout_s)
        )

    def is_busy(self) -> bool:
        """Whether the drive reports it is still moving."""
        return self._executor.read(lambda: bool(self._core.deviceBusy(self._label)))

    def limits_um(self) -> Limits | None:
        """The configured soft limits; ``None`` when none are configured."""
        limits = self._safety.z_soft_limits_um
        return None if limits is None else Limits(*limits)


class MMCamera:
    """``Camera`` over the core's current camera.

    MMCore's camera calls act on its *current* camera, not on a label.
    Switching cameras is not a capability in M1, so every call first checks
    that the current camera is still ``label`` and refuses otherwise, rather
    than silently snapping another camera.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        executor: Executor,
        pixel_sizes_um: Mapping[str, float],
        objective_label: Callable[[], str | None],
    ) -> None:
        self._core = core
        self._label = label
        self._executor = executor
        self._pixel_sizes_um = pixel_sizes_um
        self._objective_label = objective_label

    def _check_current(self) -> None:
        current = self._core.getCameraDevice()
        if current != self._label:
            raise HardwareError(
                f"camera {self._label!r} is not the core's current camera "
                f"({current!r}); switching cameras is not supported in M1 — "
                f"set Core-Camera in the configuration or assign the camera "
                f"role in the profile under [roles.assign]."
            )

    def snap(self) -> np.ndarray:
        """One frame, exactly as MMCore returns it (no copy, no flip)."""

        def action() -> np.ndarray:
            self._check_current()
            self._core.snapImage()
            return self._core.getImage()

        return self._executor.read(action)

    def exposure_ms(self) -> float:
        """The current exposure time."""

        def action() -> float:
            self._check_current()
            return float(self._core.getExposure())

        return self._executor.read(action)

    def set_exposure_ms(self, value_ms: float) -> float:
        """Set the exposure; return the camera's readback (it may clamp)."""
        value_ms = float(value_ms)

        def action() -> float:
            self._core.setExposure(value_ms)
            return float(self._core.getExposure())

        def checked() -> float:
            self._check_current()
            return self._executor.do(
                f"camera: exposure {value_ms} ms", action, dry_result=value_ms
            )

        return self._executor.read(checked)

    def image_shape(self) -> tuple[int, int]:
        """``(height, width)`` of the frames ``snap`` returns."""

        def action() -> tuple[int, int]:
            self._check_current()
            return (int(self._core.getImageHeight()), int(self._core.getImageWidth()))

        return self._executor.read(action)

    def bit_depth(self) -> int:
        """Significant bits per pixel."""

        def action() -> int:
            self._check_current()
            return int(self._core.getImageBitDepth())

        return self._executor.read(action)

    def pixel_size_um(self) -> float:
        """MMCore's calibrated value, else the profile's for the objective, else ``0.0``."""

        def action() -> float:
            measured = float(self._core.getPixelSizeUm())
            if measured > 0:
                return measured
            return float(self._pixel_sizes_um.get(self._objective_label() or "", 0.0))

        return self._executor.read(action)


class MMShutter:
    """``Shutter`` over one Micro-Manager shutter device, plus the core's auto-shutter."""

    def __init__(self, core: CMMCorePlus, label: str, executor: Executor) -> None:
        self._core = core
        self._label = label
        self._executor = executor

    def is_open(self) -> bool:
        """Whether the shutter reports open."""
        return self._executor.read(lambda: bool(self._core.getShutterOpen(self._label)))

    def set_open(self, open_: bool) -> bool:
        """Open or close; return the readback."""
        open_ = bool(open_)

        def action() -> bool:
            self._core.setShutterOpen(self._label, open_)
            return bool(self._core.getShutterOpen(self._label))

        word = "open" if open_ else "close"
        return self._executor.do(f"shutter: {word}", action, dry_result=open_)

    def auto_shutter(self) -> bool:
        """Whether the core opens the shutter around each exposure."""
        return self._executor.read(lambda: bool(self._core.getAutoShutter()))

    def set_auto_shutter(self, on: bool) -> bool:
        """Switch auto-shutter; return the readback."""
        on = bool(on)

        def action() -> bool:
            self._core.setAutoShutter(on)
            return bool(self._core.getAutoShutter())

        word = "on" if on else "off"
        return self._executor.do(f"shutter: auto_shutter {word}", action, dry_result=on)


class MMProperties:
    """``Properties`` over every loaded device of a core."""

    def __init__(self, core: CMMCorePlus, executor: Executor) -> None:
        self._core = core
        self._executor = executor

    def devices(self) -> list[str]:
        """Loaded device labels, without the ``Core`` pseudo-device."""
        return self._executor.read(
            lambda: [d for d in self._core.getLoadedDevices() if d != "Core"]
        )

    def _info(self, device: str, name: str) -> PropertyInfo:
        core = self._core
        has_limits = bool(core.hasPropertyLimits(device, name))
        return PropertyInfo(
            device=device,
            name=name,
            value=str(core.getProperty(device, name)),
            read_only=bool(core.isPropertyReadOnly(device, name)),
            allowed=tuple(str(v) for v in core.getAllowedPropertyValues(device, name)),
            lower=float(core.getPropertyLowerLimit(device, name))
            if has_limits
            else None,
            upper=float(core.getPropertyUpperLimit(device, name))
            if has_limits
            else None,
        )

    def describe(self, device: str) -> list[PropertyInfo]:
        """Every property of one device, with its constraints."""
        return self._executor.read(
            lambda: [
                self._info(device, n) for n in self._core.getDevicePropertyNames(device)
            ]
        )

    def get(self, device: str, name: str) -> str:
        """One property value, as the adapter reports it."""
        return self._executor.read(lambda: str(self._core.getProperty(device, name)))

    def set(self, device: str, name: str, value: str | float | int) -> str:
        """Set one property; return the readback.

        Raises:
            HardwareError: The property is read-only (checked before the call,
                so dry-run refuses it too).
        """

        def action() -> str:
            self._core.setProperty(device, name, value)
            return str(self._core.getProperty(device, name))

        def checked() -> str:
            if self._core.isPropertyReadOnly(device, name):
                raise HardwareError(
                    f"{device}.{name} is read-only; it reports state, it cannot be "
                    f"set. See `describe({device!r})` for the settable properties."
                )
            return self._executor.do(
                f"property: {device}.{name}={value}", action, dry_result=str(value)
            )

        return self._executor.read(checked)
