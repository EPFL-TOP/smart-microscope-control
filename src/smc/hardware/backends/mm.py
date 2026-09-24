"""Micro-Manager implementations of the capabilities (design §6).

Each class wraps one device label of a ``CMMCorePlus`` and nothing else: it
does not choose the device (roles do, #6) and does not own the core (the
facade does, #8). Every action goes through the microscope's ``Executor``,
so all of them share one lock and honour dry-run.

Hardware behaviour encoded here:

* **Explicit labels.** Stage and shutter calls pass the device label to
  MMCore instead of relying on the core's "current" XY/focus/shutter slot, so
  a stand with several stages (the Ti2 has four ``XYStage`` devices, FM-13)
  moves the one the role names.
* **Readback.** A mutation returns what the device reports afterwards, not
  what was asked: stages round to their step size, cameras clamp exposure.
* **One lock section per action** (design §13). A move checks, commands,
  waits and reads back while it holds the microscope lock, so its target
  and its readback are its own. Reads, ``stop()`` and closing a shutter
  skip the lock: MMCore serialises calls to one adapter, so they are safe
  during a move, and a stop gets through a plate traverse. The wait's
  deadline defaults to the core timeout, which the facade raises to 60 s
  because a plate traverse outlasts MMCore's 5 s default (FM-10); a wait
  that gives up stops the stage first (FM-17).
* **Pixel size is measured or unknown.** MMCore's calibrated value when it
  has one, else the profile's value for the current objective, else ``0.0``.
* **Dry-run reads real state.** Safety checks and the position a relative
  move starts from use live reads; only the mutation is skipped.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from smc.hardware.capabilities import XY, Limits, PropertyInfo
from smc.hardware.errors import HardwareError
from smc.hardware.safety import Motion, Step

if TYPE_CHECKING:
    import numpy as np
    from pymmcore_plus import CMMCorePlus

    from smc.hardware.safety import Executor, Safety

__all__ = ["MMCamera", "MMProperties", "MMShutter", "MMXYStage", "MMZStage"]

logger = logging.getLogger("smc.hardware.backends.mm")


def _motion(core: CMMCorePlus, label: str, name: str, *, stoppable: bool) -> Motion:
    def stop() -> None:
        core.stop(label)

    return Motion(
        device=label,
        name=name,
        is_busy=lambda: bool(core.deviceBusy(label)),
        stop=stop if stoppable else None,
    )


def _core_timeout(core: CMMCorePlus) -> Callable[[], float]:
    """The core timeout in seconds, resolved inside the action (design §2, §13)."""
    return lambda: core.getTimeoutMs() / 1000.0


class MMXYStage:
    """``XYStage`` over one Micro-Manager XY stage device."""

    def __init__(
        self, core: CMMCorePlus, label: str, executor: Executor, safety: Safety
    ) -> None:
        self._core = core
        self._label = label
        self._executor = executor
        self._safety = safety
        self._motion = _motion(core, label, f"xy_stage {label}", stoppable=True)

    def _read_position(self) -> XY:
        return XY(
            float(self._core.getXPosition(self._label)),
            float(self._core.getYPosition(self._label)),
        )

    def position_um(self) -> XY:
        """Where the stage reports it is."""
        return self._executor.read(self._read_position)

    def _step(self, log: str, x_um: float, y_um: float) -> Step[XY]:
        return Step(
            log=log,
            send=lambda: self._core.setXYPosition(self._label, x_um, y_um),
            readback=self._read_position,
            dry_result=XY(x_um, y_um),
        )

    def _move(self, description: str, step: Callable[[], Step[XY]], wait: bool) -> XY:
        return self._executor.do(
            description,
            step,
            motion=self._motion,
            wait=wait,
            timeout_s=_core_timeout(self._core),
        )

    def move_to_um(self, x_um: float, y_um: float, *, wait: bool = True) -> XY:
        """Move to an absolute position (soft limits apply, no jog guard).

        With ``wait=False`` the return value is the position read right after
        the command, not where the stage will land; call ``wait()`` then
        ``position_um()`` for that.
        """
        x_um, y_um = float(x_um), float(y_um)
        self._safety.check_xy_target_um(x_um, y_um)
        description = f"xy_stage: move_to ({x_um}, {y_um}) µm"
        return self._move(
            description, lambda: self._step(description, x_um, y_um), wait
        )

    def move_by_um(
        self, dx_um: float, dy_um: float, *, wait: bool = True, force: bool = False
    ) -> XY:
        """Move relative to the live position; jog-guarded unless ``force``.

        The target is computed from a position read inside the action, after
        the guard, and moved to absolutely: the soft-limit check sees the real
        destination, and no other caller can move the stage in between.
        """
        dx_um, dy_um = float(dx_um), float(dy_um)
        self._safety.check_jog_um(dx_um, dy_um, force=force)

        def relative() -> Step[XY]:
            here = self._read_position()
            x_um, y_um = here.x_um + dx_um, here.y_um + dy_um
            self._safety.check_xy_target_um(x_um, y_um)
            return self._step(
                f"xy_stage: move_by ({dx_um}, {dy_um}) µm to ({x_um}, {y_um}) µm",
                x_um,
                y_um,
            )

        return self._move(f"xy_stage: move_by ({dx_um}, {dy_um}) µm", relative, wait)

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until idle; ``DeviceTimeoutError`` after ``timeout_s`` (default: core timeout).

        The thread that started the move stops the stage before a timeout or
        an interrupt propagates; a move that was stopped raises
        ``MotionStoppedError``.
        """
        self._executor.wait(
            self._motion, _core_timeout(self._core) if timeout_s is None else timeout_s
        )

    def is_busy(self) -> bool:
        """Whether the stage reports it is still moving."""
        return self._executor.read(lambda: bool(self._core.deviceBusy(self._label)))

    def stop(self) -> None:
        """Stop the stage now, without waiting for the microscope lock (§13)."""
        self._executor.stop(self._motion)

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
        self._motion = _motion(core, label, f"z {label}", stoppable=True)

    def _read_position(self) -> float:
        return float(self._core.getPosition(self._label))

    def position_um(self) -> float:
        """Where the drive reports it is."""
        return self._executor.read(self._read_position)

    def _step(self, log: str, z_um: float) -> Step[float]:
        return Step(
            log=log,
            send=lambda: self._core.setPosition(self._label, z_um),
            readback=self._read_position,
            dry_result=z_um,
        )

    def _move(
        self, description: str, step: Callable[[], Step[float]], wait: bool
    ) -> float:
        return self._executor.do(
            description,
            step,
            motion=self._motion,
            wait=wait,
            timeout_s=_core_timeout(self._core),
        )

    def move_to_um(self, z_um: float, *, wait: bool = True) -> float:
        """Move to an absolute position (soft limits apply).

        With ``wait=False`` the return value is the position read right after
        the command, not where the stage will land; call ``wait()`` then
        ``position_um()`` for that.
        """
        z_um = float(z_um)
        self._safety.check_z_target_um(z_um)
        description = f"z: move_to {z_um} µm"
        return self._move(description, lambda: self._step(description, z_um), wait)

    def move_by_um(self, dz_um: float, *, wait: bool = True) -> float:
        """Move relative to the live position (soft limits apply to the target).

        The anchor is read inside the action, as for ``XYStage.move_by_um``.
        """
        dz_um = float(dz_um)

        def relative() -> Step[float]:
            z_um = self._read_position() + dz_um
            self._safety.check_z_target_um(z_um)
            return self._step(f"z: move_by {dz_um} µm to {z_um} µm", z_um)

        return self._move(f"z: move_by {dz_um} µm", relative, wait)

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until idle; ``DeviceTimeoutError`` after ``timeout_s`` (default: core timeout).

        Same contract as ``XYStage.wait``.
        """
        self._executor.wait(
            self._motion, _core_timeout(self._core) if timeout_s is None else timeout_s
        )

    def is_busy(self) -> bool:
        """Whether the drive reports it is still moving."""
        return self._executor.read(lambda: bool(self._core.deviceBusy(self._label)))

    def stop(self) -> None:
        """Stop the drive now, without waiting for the microscope lock (§13)."""
        self._executor.stop(self._motion)

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
        """One frame, exactly as MMCore returns it (no copy, no flip).

        An acquisition (§13): refused while halted or while a device the layer
        moved is still moving, because a frame taken mid-move is a wrong
        result.

        Unbounded (FM-15): ``snapImage`` has no timeout of its own, and a thread
        inside a driver call cannot be cancelled. It runs under the microscope
        lock, so a hung camera driver holds it; other actions then fail after
        the lock timeout with ``MicroscopeBusyError`` naming "camera: snap"
        instead of hanging, while reads, ``stop()`` and closing the shutter
        still get through.
        """

        def action() -> np.ndarray:
            self._check_current()
            self._core.snapImage()
            return self._core.getImage()

        return self._executor.acquire("camera: snap", action)

    def exposure_ms(self) -> float:
        """The current exposure time."""

        def action() -> float:
            self._check_current()
            return float(self._core.getExposure())

        return self._executor.read(action)

    def set_exposure_ms(self, value_ms: float) -> float:
        """Set the exposure; return the camera's readback (it may clamp).

        Raises:
            ValueError: ``value_ms`` is negative or not finite; it never reaches
                the driver, whose handling of such values is undefined.
        """
        value_ms = float(value_ms)
        if not math.isfinite(value_ms) or value_ms < 0:
            raise ValueError(f"exposure must be finite and >= 0 ms, got {value_ms!r}")
        description = f"camera: exposure {value_ms} ms"

        def step() -> Step[float]:
            # Checked inside the action, in dry-run too.
            self._check_current()
            return Step(
                log=description,
                send=lambda: self._core.setExposure(value_ms),
                readback=lambda: float(self._core.getExposure()),
                dry_result=value_ms,
            )

        return self._executor.do(description, step)

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
        """MMCore's calibrated value, else the profile's for the objective, else ``0.0``.

        MMCore's value already includes binning; the profile map is for
        binning 1, so the fallback is multiplied by the current binning. A
        binning that cannot be read, or a map entry that is not a positive
        finite number, gives ``0.0`` (unknown) with a warning rather than a
        pixel size that looks measured but is wrong. A device that fails to
        answer raises, as every reader does (design §3); the facade's
        ``state()`` is the tolerant layer.
        """

        def action() -> float:
            # MMCore's value is for its current camera and binning; for another
            # camera it would be a measurement of the wrong device.
            self._check_current()
            measured = float(self._core.getPixelSizeUm())
            if measured > 0:
                return measured
            objective = self._objective_label()
            if objective is None or objective not in self._pixel_sizes_um:
                return 0.0
            at_bin_1 = float(self._pixel_sizes_um[objective])
            if not (math.isfinite(at_bin_1) and at_bin_1 > 0):
                logger.warning(
                    "pixel size for objective %r in the profile is %r, not a "
                    "positive number; reporting 0.0 (unknown). Fix it under "
                    "[camera] pixel_size_um in the profile.",
                    objective,
                    at_bin_1,
                )
                return 0.0
            binning = self._binning()
            if binning is None:
                return 0.0
            return at_bin_1 * binning

        return self._executor.read(action)

    def _binning(self) -> int | None:
        """The camera's square binning factor, or ``None`` (with a warning) if unreadable.

        Adapters spell it ``"2"`` or ``"2x2"``; a non-square binning has no
        single pixel size and counts as unreadable.
        """
        raw = ""
        if self._core.hasProperty(self._label, "Binning"):
            raw = str(self._core.getProperty(self._label, "Binning"))
        parts = raw.lower().replace(" ", "").split("x")
        square = len(parts) <= 2 and len(set(parts)) == 1
        if square and parts[0].isdigit() and int(parts[0]) >= 1:
            return int(parts[0])
        logger.warning(
            "camera %r binning %r cannot be read as one integer factor; the "
            "profile pixel size cannot be scaled, reporting 0.0 (unknown).",
            self._label,
            raw,
        )
        return None


class MMShutter:
    """``Shutter`` over one Micro-Manager shutter device, plus the core's auto-shutter."""

    def __init__(self, core: CMMCorePlus, label: str, executor: Executor) -> None:
        self._core = core
        self._label = label
        self._executor = executor

    def _read_open(self) -> bool:
        return bool(self._core.getShutterOpen(self._label))

    def is_open(self) -> bool:
        """Whether the shutter reports open."""
        return self._executor.read(self._read_open)

    def set_open(self, open_: bool) -> bool:
        """Open or close; return the readback.

        Closing is a safe call (§13): it never waits for the microscope lock
        and is never refused, halted, moving or in dry-run, so the light can
        always be cut. Opening is an action like any other.
        """
        if not open_:

            def close() -> bool:
                self._core.setShutterOpen(self._label, False)
                return self._read_open()

            return self._executor.safe("shutter: close", close)
        return self._executor.do(
            "shutter: open",
            Step(
                log="shutter: open",
                send=lambda: self._core.setShutterOpen(self._label, True),
                readback=self._read_open,
                dry_result=True,
            ),
        )

    def auto_shutter(self) -> bool:
        """Whether the core opens the shutter around each exposure."""
        return self._executor.read(lambda: bool(self._core.getAutoShutter()))

    def set_auto_shutter(self, on: bool) -> bool:
        """Switch auto-shutter; return the readback."""
        on = bool(on)
        description = f"shutter: auto_shutter {'on' if on else 'off'}"
        return self._executor.do(
            description,
            Step(
                log=description,
                send=lambda: self._core.setAutoShutter(on),
                readback=lambda: bool(self._core.getAutoShutter()),
                dry_result=on,
            ),
        )


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

        On a ``State`` device (a turret, a filter wheel, a light path) the set
        can start a movement, so it is a motion (§13): it waits inside the
        action for the device to report idle before reading back, with the
        core timeout as the deadline. A property of a stage is not a motion:
        moving a stage through ``Properties`` bypasses every guard.

        Raises:
            HardwareError: The property is read-only (checked before the call,
                so dry-run refuses it too).
        """
        if self._core.isPropertyReadOnly(device, name):
            raise HardwareError(
                f"{device}.{name} is read-only; it reports state, it cannot be "
                f"set. See `describe({device!r})` for the settable properties."
            )
        description = f"property: {device}.{name}={value}"
        motion = self._motion_for(device)
        return self._executor.do(
            description,
            Step(
                log=description,
                send=lambda: self._core.setProperty(device, name, value),
                readback=lambda: str(self._core.getProperty(device, name)),
                dry_result=str(value),
            ),
            motion=motion,
            timeout_s=_core_timeout(self._core),
        )

    def _motion_for(self, device: str) -> Motion | None:
        """The ``Motion`` a set on ``device`` starts, or ``None`` if it moves nothing we track."""
        from pymmcore_plus import DeviceType

        try:
            kind = DeviceType(self._core.getDeviceType(device))
        except ValueError:
            return None  # a type this pymmcore-plus does not know
        if kind != DeviceType.State:
            return None
        return _motion(self._core, device, device, stoppable=False)
