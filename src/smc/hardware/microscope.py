"""The ``Microscope`` facade: one stand, its roles, its capabilities (design §7, §13).

This is what tools and user interfaces hold. It opens one core from a
profile, decides which device fills each role, and hands out capabilities
that all share one ``Executor`` (one lock, one dry-run flag, one halt) and
one ``Safety`` built from the profile. A tool never builds a backend class
itself, so it cannot end up with a stage that skips the guards or a second
lock that lets two actions interleave.

Hardware behaviour encoded here:

* **The core timeout comes from the profile** (FM-10): MMCore's 5 s default
  is shorter than a plate traverse, and a move gives up at the core timeout.
  The lock timeout follows it (FM-15), so a caller behind a hung ``snap``
  fails with a diagnosis after the same bound.
* **Every role warning is logged** (FM-13): a stand where several devices
  share a type is where a wrong pick hides, and a ``.cfg`` that names a TIRF
  positioner as the XY stage is corrected with a warning, not in silence.
* **One connection per stand.** Nikon and Zeiss stands accept one client at
  a time, so ``close()`` unloads every device, a failed ``open()`` releases
  the core it opened, and closing twice is harmless.
* **The emergency stop halts first** (§13): ``stop()`` refuses every new
  action before it sends the stops, so a plugin loop cannot start its next
  move between the stop of one stage and the next.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar, cast

from smc.hardware.backends.mm import (
    MMCamera,
    MMProperties,
    MMShutter,
    MMXYStage,
    MMZStage,
)
from smc.hardware.capabilities import XY, Camera, Properties, Shutter, XYStage, ZStage
from smc.hardware.core import close_core, open_core
from smc.hardware.errors import (
    CapabilityMissingError,
    HardwareError,
    MicroscopeBusyError,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import DeviceInfo, Role, RoleMap, core_roles, devices_from_core
from smc.hardware.roles import resolve as resolve_roles
from smc.hardware.safety import Executor, Safety

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = ["CAPABILITY_ROLES", "Microscope", "MicroscopeState"]

log = logging.getLogger("smc.hardware.microscope")

T = TypeVar("T")
R = TypeVar("R")

#: The roles each capability needs, in the order ``available()`` lists them.
#: ``Properties`` needs none: it reaches every loaded device.
CAPABILITY_ROLES: dict[type, tuple[Role, ...]] = {
    XYStage: (Role.xy_stage,),
    ZStage: (Role.focus,),
    Camera: (Role.camera,),
    Shutter: (Role.shutter,),
    Properties: (),
}

#: The stages the emergency stop reaches, in the order it stops them.
_STAGES: tuple[tuple[type, Role], ...] = (
    (XYStage, Role.xy_stage),
    (ZStage, Role.focus),
)


@dataclass
class MicroscopeState:
    """A snapshot of the stand, as tolerant as a status bar must be.

    A field is ``None`` when its capability is missing or its read failed;
    ``errors`` says which failed and why (``"z: RuntimeError: ..."``), so one
    unplugged device never hides the others.
    """

    xy: XY | None = None
    z_um: float | None = None
    exposure_ms: float | None = None
    image_shape: tuple[int, int] | None = None
    pixel_size_um: float | None = None
    shutter_open: bool | None = None
    moving: tuple[str, ...] = ()
    halted: bool = False
    roles: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class Microscope:
    """One stand: its profile, its roles, and the capabilities that fill them.

    Build it with :meth:`open` (from a profile) or :meth:`from_core` (tests,
    fakes). Capabilities are built once, on first request, and cached, so
    every caller of ``require(XYStage)`` shares one stage and one lock.

    ``core`` stays public for the CLI and the tests; a plugin that touches it
    bypasses every guard and fails review (design §12).
    """

    profile: Profile
    roles: RoleMap
    devices: list[DeviceInfo]
    core: CMMCorePlus

    def __init__(
        self, core: CMMCorePlus, profile: Profile, *, dry_run: bool = False
    ) -> None:
        """Wire a core that is already loaded; prefer :meth:`open` or :meth:`from_core`."""
        self.core = core
        self.profile = profile
        timeout_ms = profile.micromanager.device_timeout_ms
        # FM-10: a move's deadline is the core timeout, resolved per action.
        core.setTimeoutMs(timeout_ms)
        self.devices = devices_from_core(core)
        self.roles = resolve_roles(
            self.devices,
            core_roles=core_roles(core),
            overrides=profile.roles.assign,
            exclusions=profile.roles.exclude,
        )
        for warning in self.roles.warnings:
            log.warning("%s", warning)
        safety = profile.safety
        self._safety = Safety(
            max_jog_um=safety.max_jog_um,
            z_soft_limits_um=safety.z_soft_limits_um,
            xy_soft_limits_um=safety.xy_soft_limits_um,
        )
        self._lock = threading.RLock()
        # FM-15: waiting for the lock is bounded like a device call.
        self._lock_timeout_s = timeout_ms / 1000
        self._executor = Executor(
            dry_run=dry_run,
            lock=self._lock,
            logger=log,
            lock_timeout_s=self._lock_timeout_s,
        )
        self._capabilities: dict[type, object] = {}
        # Re-entrant: the emergency stop takes it from inside a lookup (FM-70).
        self._capabilities_lock = threading.RLock()
        self._closed = False

    @classmethod
    def open(
        cls, profile: Profile | str | Path = "demo", *, dry_run: bool = False
    ) -> Microscope:
        """Load a profile, open its Micro-Manager configuration, resolve the roles.

        ``"demo"`` needs no file: :meth:`Profile.load` falls back to
        :meth:`Profile.demo`.

        Raises:
            ProfileError: The profile cannot be found or does not validate.
            CoreError: The configuration cannot be loaded.
        """
        loaded = profile if isinstance(profile, Profile) else Profile.load(profile)
        core = open_core(loaded.config_path())
        try:
            microscope = cls.from_core(core, loaded, dry_run=dry_run)
        except BaseException:
            # One connection per stand: a half-built facade must not keep it.
            close_core(core)
            raise
        log.info(
            "%s: opened, %d devices%s",
            loaded.microscope.name,
            len(microscope.devices),
            " (dry-run)" if dry_run else "",
        )
        return microscope

    @classmethod
    def from_core(
        cls, core: CMMCorePlus, profile: Profile, *, dry_run: bool = False
    ) -> Microscope:
        """Wrap a core that is already loaded (tests, a ``FakeCore``); ``open`` minus loading.

        The facade then owns the core: :meth:`close` unloads its devices.
        """
        return cls(core, profile, dry_run=dry_run)

    @property
    def dry_run(self) -> bool:
        """Whether mutations are logged and skipped; fixed when the facade is built."""
        return self._executor.dry_run

    # --- capabilities ------------------------------------------------------

    def has(self, capability: type[T]) -> bool:
        """Whether a device fills every role ``capability`` needs (or it was overridden)."""
        return self._lookup(capability) is not None

    def get(self, capability: type[T]) -> T | None:
        """The capability, built once and cached; ``None`` when no device fills its role."""
        return cast("T | None", self._lookup(capability))

    def require(self, capability: type[T]) -> T:
        """The capability, or an error that names the missing role and what is available.

        Raises:
            CapabilityMissingError: No loaded device fills the role; the
                message points at ``[roles.assign]`` in the profile.
        """
        found = self._lookup(capability)
        if found is None:
            missing = next(
                (
                    r
                    for r in CAPABILITY_ROLES.get(capability, ())
                    if r not in self.roles.assigned
                ),
                None,
            )
            raise CapabilityMissingError(
                capability.__name__,
                missing,
                tuple(role.value for role in self.roles.assigned),
            )
        return cast("T", found)

    def available(self) -> list[type]:
        """The capabilities this stand offers, in ``CAPABILITY_ROLES`` order, then overrides."""
        known = [cap for cap in CAPABILITY_ROLES if self._lookup(cap) is not None]
        with self._capabilities_lock:
            extra = [cap for cap in self._capabilities if cap not in CAPABILITY_ROLES]
        return known + extra

    def override(self, capability: type[T], implementation: T) -> None:
        """Replace a capability for the rest of this facade's life; a testing hook.

        The synthetic camera (#10) goes in through here. The implementation
        bypasses the ``Executor``, so it takes no lock and ignores dry-run
        and the halt unless it implements them itself.

        Raises:
            TypeError: ``implementation`` lacks a method of the capability's
                Protocol, which would otherwise surface at the first call.
        """
        if capability in CAPABILITY_ROLES and not isinstance(
            implementation, capability
        ):
            raise TypeError(
                f"{type(implementation).__name__} does not implement "
                f"{capability.__name__}: it lacks one of its methods"
            )
        with self._capabilities_lock:
            self._capabilities[capability] = implementation
        log.info(
            "%s: overridden by %s", capability.__name__, type(implementation).__name__
        )

    def _lookup(self, capability: type) -> object | None:
        """The cached capability, built on first request; ``None`` when a role is unfilled.

        The untyped core of ``has``/``get``/``require`` and of the stop path,
        which iterates plain ``type`` keys; the public methods cast.

        The cache lock is re-entrant: a Ctrl-C handler that calls ``stop()``
        or ``close()`` can run on a thread that is inside this method, and
        the stop path looks the stages up here (FM-70). Such a handler may
        also build the capability this call is building; the first one cached
        is kept, so a facade never hands out two stages.
        """
        with self._capabilities_lock:
            found = self._capabilities.get(capability)
            if found is None:
                built = self._build(capability)
                if built is not None:
                    found = self._capabilities.setdefault(capability, built)
            return found

    def _build(self, capability: type) -> object | None:
        """The Micro-Manager implementation of ``capability``; no device is contacted."""
        get = self.roles.get
        core, executor, safety = self.core, self._executor, self._safety
        if capability is Properties:
            return MMProperties(core, executor)
        if capability is XYStage and (label := get(Role.xy_stage)) is not None:
            return MMXYStage(core, label, executor, safety)
        if capability is ZStage and (label := get(Role.focus)) is not None:
            return MMZStage(core, label, executor, safety)
        if capability is Shutter and (label := get(Role.shutter)) is not None:
            return MMShutter(core, label, executor)
        if capability is Camera and (label := get(Role.camera)) is not None:
            return MMCamera(
                core,
                label,
                executor,
                self.profile.camera.pixel_size_um,
                self._objective_label(),
            )
        return None

    def _objective_label(self) -> Callable[[], str | None]:
        """The current objective's name, for the profile's pixel-size map."""
        turret = self.roles.get(Role.objective_turret)
        if turret is None:
            return lambda: None
        core = self.core
        return lambda: str(core.getStateLabel(turret))

    # --- state -------------------------------------------------------------

    def state(self) -> MicroscopeState:
        """Read everything a status line shows; a failed read is an entry in ``errors``.

        A read, like every capability read (§13): it takes no lock, so it
        runs during a move, while halted and in dry-run.
        """
        state = MicroscopeState(
            roles={role.value: label for role, label in self.roles.assigned.items()}
        )
        errors = state.errors
        xy = cast("XYStage | None", self._lookup(XYStage))
        if xy is not None:
            state.xy = _read("xy", xy.position_um, errors)
        z = cast("ZStage | None", self._lookup(ZStage))
        if z is not None:
            state.z_um = _read("z", z.position_um, errors)
        camera = cast("Camera | None", self._lookup(Camera))
        if camera is not None:
            state.exposure_ms = _read("exposure", camera.exposure_ms, errors)
            state.image_shape = _read("image_shape", camera.image_shape, errors)
            state.pixel_size_um = _read("pixel_size", camera.pixel_size_um, errors)
        shutter = cast("Shutter | None", self._lookup(Shutter))
        if shutter is not None:
            state.shutter_open = _read("shutter", shutter.is_open, errors)
        # Read after the devices: "moving" and "halted" are then at least as
        # recent as the positions shown next to them.
        state.moving = self._executor.moving()
        state.halted = self._executor.halted
        return state

    def describe(self) -> str:
        """One status line: ``XY (x, y) µm | Z z µm | exposure ms | shutter open``.

        A capability the stand lacks is left out; one whose read failed shows
        ``?``. ``| moving: …`` and ``| HALTED`` are added when they apply.
        """
        state = self.state()
        parts: list[str] = []
        if self._lookup(XYStage) is not None:
            parts.append(
                "XY ?"
                if state.xy is None
                else f"XY ({state.xy.x_um:.1f}, {state.xy.y_um:.1f}) µm"
            )
        if self._lookup(ZStage) is not None:
            parts.append("Z ?" if state.z_um is None else f"Z {state.z_um:.2f} µm")
        if self._lookup(Camera) is not None:
            parts.append(
                "exposure ?"
                if state.exposure_ms is None
                else f"{state.exposure_ms:g} ms"
            )
        if self._lookup(Shutter) is not None:
            if state.shutter_open is None:
                parts.append("shutter ?")
            else:
                parts.append("shutter open" if state.shutter_open else "shutter closed")
        if state.moving:
            parts.append("moving: " + ", ".join(state.moving))
        if state.halted:
            parts.append("HALTED")
        return " | ".join(parts) or "no stage, camera or shutter"

    # --- stop, resume, close -----------------------------------------------

    def stop(self) -> None:
        """The stand's emergency stop: halt, then stop every stage (§13).

        The halt comes first, so no action starts once the stops are under
        way; every stage is stopped even when an earlier one fails. The stops
        never wait for the microscope lock, so they get through a move that
        holds it. Every action then raises ``MicroscopeHaltedError`` until
        :meth:`resume`; reads and closing a shutter still run.

        Raises:
            HardwareError: One or more stages did not take the stop; the
                message names each, and the microscope stays halted.
        """
        self._executor.halt()
        failures = self._stop_stages()
        if failures:
            raise HardwareError(
                f"the emergency stop failed for {'; '.join(failures)}. The "
                f"microscope stays halted: stop the stage at the stand if it "
                f"still moves, then call `resume()`"
            )

    def resume(self) -> None:
        """Clear the halt and forget every motion still tracked; starts nothing (§13)."""
        self._executor.resume()

    def close(self) -> None:
        """Stop what still moves, then unload every device; safe to call twice.

        The stand accepts one connection at a time, so the devices are
        released here. A stop that fails is logged and does not keep the
        stand connected. Unloading waits for the microscope lock, so it never
        pulls a device from under an action in another thread; that wait is
        bounded like any action's.

        Raises:
            MicroscopeBusyError: Another call held the microscope lock past
                the lock timeout (a hung driver); nothing was unloaded, and
                ``close()`` may be called again.
            HardwareError: ``close()`` was called from inside an action of
                this thread (a Ctrl-C handler that runs during a move, a
                callback inside a snap). The stages were stopped if anything
                moved, but nothing was unloaded: the action still uses the
                devices. Call ``close()`` again once it has returned.
        """
        if self._closed:
            return
        if _owns(self._lock):
            self._stop_if_moving()
            raise HardwareError(
                "close() was called from inside an action on this thread, which "
                "still uses the devices; the stages were stopped if anything "
                "moved, and nothing was unloaded. Call close() again once the "
                "action has returned (after the `with` block, or in a `finally`)"
            )
        acquired = stopped = False
        # Nothing between acquire() and the try (FM-65); the finally also asks
        # the lock, in case an interrupt landed before `acquired` was set.
        try:
            acquired = self._lock.acquire(blocking=False)
            if not acquired:
                # Another thread's action holds the lock. If it is waiting for
                # a motion, the stop ends it; otherwise close would wait for
                # the whole traverse.
                started = time.monotonic()
                stopped = self._stop_if_moving()
                acquired = self._lock.acquire(timeout=self._lock_timeout_s)
                if not acquired:
                    # Only this wait is known, not when the holder started.
                    raise MicroscopeBusyError(
                        "another call", time.monotonic() - started, at_least=True
                    )
            if not stopped:
                self._stop_if_moving()
            try:
                self.core.unloadAllDevices()
            except Exception as exc:
                log.error(
                    "%s: unloading the devices failed (%r); the stand may still "
                    "be connected, so restart the program before another one "
                    "connects to it",
                    self.profile.microscope.name,
                    exc,
                )
            self._closed = True
        finally:
            if acquired or _owns(self._lock):
                self._lock.release()
        log.info("%s: closed", self.profile.microscope.name)

    def __enter__(self) -> Microscope:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _stop_if_moving(self) -> bool:
        """For ``close()``: stop every stage if anything still moves; whether it did.

        A failed stop is logged, never raised: the devices are unloaded
        anyway, and a stop that did not take is a finding for the log.
        """
        moving = self._executor.moving()
        if not moving:
            return False
        failures = self._stop_stages()
        # Logged after the stops were sent (FM-62).
        log.warning(
            "close: %s still moving; the stages were stopped", ", ".join(moving)
        )
        for failure in failures:
            log.error("close: the stop failed for %s", failure)
        return True

    def _stop_stages(self) -> list[str]:
        """Stop each stage the stand has, past any failure; return one note per failure.

        An interrupt (a second Ctrl-C) during one stop does not skip the
        next: it is re-raised once every stage has had its stop.
        """
        failures: list[str] = []
        interrupted: BaseException | None = None
        for capability, role in _STAGES:
            stage = cast("XYStage | ZStage | None", self._lookup(capability))
            if stage is None:
                continue
            try:
                stage.stop()
            except Exception as exc:
                label = self.roles.get(role)
                name = capability.__name__ + (f" {label!r}" if label else "")
                failures.append(f"{name} ({exc!r})")
            except BaseException as exc:
                interrupted = interrupted or exc
        if interrupted is not None:
            raise interrupted
        return failures


def _owns(lock: threading.RLock) -> bool:
    """Whether the calling thread holds ``lock`` (``RLock._is_owned``, as ``Condition`` uses)."""
    is_owned = getattr(lock, "_is_owned", None)
    return bool(is_owned()) if is_owned is not None else False


def _read(name: str, read: Callable[[], R], errors: list[str]) -> R | None:
    """``read()``, or ``None`` with ``"<name>: <error>"`` added to ``errors``."""
    try:
        return read()
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        log.debug("state: %s read failed", name, exc_info=True)
        return None
