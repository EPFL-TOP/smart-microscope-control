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
* **An assumed Z jog limit is logged** (#84): a profile that sets no
  ``[safety] max_z_jog_um`` gets ``DEFAULT_MAX_Z_JOG_UM``, and one WARNING at
  open says the number is a default and not the stand's, when a focus drive
  is present.
* **One connection per stand.** Nikon and Zeiss stands accept one client at
  a time, so ``close()`` unloads every device, a failed ``open()`` releases
  the core it opened, and closing twice is harmless.
* **The emergency stop halts first and logs last** (§13): ``stop()``
  refuses every new action before it sends the stops, so a plugin loop
  cannot start its next move between the stop of one stage and the next.
  Its lines, the halt's included, are written once every stop is sent, so
  a blocked console (a QuickEdit selection on Windows) cannot hold the Z
  stop behind the line of the XY stop (FM-62). **The autofocus is switched
  off before the stages** (m2 §3.11): a Z stop sent while PFS still drives Z
  can be overridden by its next correction. Its offset device is stopped
  right after, by its role: it is a stage the layer moves.
* **The core's AutoFocus slot follows the role** (FM-87): the
  continuous-focus calls act on the slot's device, so when the role names
  another one, the slot is pointed at it once, here.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar, cast

from smc.hardware.backends.mm import (
    MMAutofocus,
    MMCamera,
    MMChannels,
    MMLightSource,
    MMObjectiveTurret,
    MMProperties,
    MMShutter,
    MMXYStage,
    MMZStage,
    ZInterlock,
    as_hardware_errors,
    autofocus_offset_motion,
)
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Channels,
    LightSource,
    ObjectiveTurret,
    Properties,
    Shutter,
    XYStage,
    ZStage,
)
from smc.hardware.channels import choose_group
from smc.hardware.core import close_core, open_core
from smc.hardware.errors import (
    CapabilityMissingError,
    HardwareError,
    MicroscopeBusyError,
    StopNote,
    StopOutcome,
    StopRefusedError,
    join_names,
    note_stop,
    stop_notes,
)
from smc.hardware.profile import Profile
from smc.hardware.roles import DeviceInfo, Role, RoleMap, core_roles, devices_from_core
from smc.hardware.roles import resolve as resolve_roles
from smc.hardware.safety import DEFAULT_MAX_Z_JOG_UM, Executor, Safety

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = ["CAPABILITY_ROLES", "Microscope", "MicroscopeState"]

log = logging.getLogger("smc.hardware.microscope")

T = TypeVar("T")
R = TypeVar("R")

#: The roles each capability needs, in the order ``available()`` lists them.
#: ``Properties`` needs none: it reaches every loaded device. ``Channels``
#: needs none either, but a config group named for channels (m2 §8.5).
CAPABILITY_ROLES: dict[type, tuple[Role, ...]] = {
    XYStage: (Role.xy_stage,),
    ZStage: (Role.focus,),
    Camera: (Role.camera,),
    Shutter: (Role.shutter,),
    Autofocus: (Role.autofocus,),
    ObjectiveTurret: (Role.objective_turret,),
    LightSource: (Role.light_source,),
    Channels: (),
    Properties: (),
}

#: Why ``require(Channels)`` fails on a stand with no channel group.
_NO_CHANNEL_GROUP = (
    "no config group is named for channels; name one in the profile's [channels] group"
)

#: What the emergency stop reaches, in the order it stops them: the
#: autofocus first, switched off, then the offset device it follows, then
#: the stages (m2 §3.11). The offset device fills a role but is no
#: capability (``None``): it is stopped by its role's label.
_STOPPED: tuple[tuple[type | None, Role], ...] = (
    (Autofocus, Role.autofocus),
    (None, Role.autofocus_offset),
    (XYStage, Role.xy_stage),
    (ZStage, Role.focus),
)

#: How the backends name each stopped stage's motion (``"z Z"``), for the
#: emergency stop's and ``close()``'s messages: the same names as their lines.
_STOP_NAMES: dict[Role, str] = {
    Role.xy_stage: "xy_stage",
    Role.focus: "z",
    Role.autofocus_offset: "autofocus_offset",
}


@dataclass
class _Stops:
    """What ``_stop_stages`` did, as text: no exception, so no traceback (FM-98).

    ``done``: ``autofocus <label> off`` and each stop the device took;
    ``refused``: each refusal the profile declares; ``failed``: each failure,
    with its error's ``repr`` (m2 §11.5).
    """

    done: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _note_still_moving(exc: BaseException | None, tracked: tuple[str, ...]) -> None:
    """Note on ``exc`` each motion in ``tracked`` that its own notes do not name.

    A Ctrl-C in a call that sends no move of its own (``wait()`` on a stage an
    earlier give-up left tracked) carries no note, and the Ctrl-C line would say
    that no move was under way while the stage travels (m2 §11.11 E6). A
    motion whose give-up already left a note is not noted again, and a name is
    noted once. A module function: it holds ``exc`` only as a parameter and
    raises nothing, so no frame on a traceback keeps an exception (FM-98).
    """
    if exc is None or not tracked:
        return
    noted = {note.motion for note in stop_notes(exc)}
    for name in tracked:
        if name not in noted:
            noted.add(name)
            note_stop(exc, StopNote(name, StopOutcome.still_moving))


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
    autofocus_engaged: bool | None = None
    autofocus_locked: bool | None = None
    #: In the offset device's own units ([autofocus] offset_unit).
    autofocus_offset_native: float | None = None
    objective: str | None = None
    #: ``None`` also when the stand matches no preset of the channel group.
    channel: str | None = None
    #: In the light source's own units.
    light_intensity_native: float | None = None
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
        self._point_autofocus_slot()
        safety = profile.safety
        # Only a stand that has a focus drive has a Z jog to guard; the
        # profile cannot know that at load time, so the warning waits for the
        # roles. The number in force is DEFAULT_MAX_Z_JOG_UM, not the stand's.
        if safety.max_z_jog_um is None and self.roles.get(Role.focus) is not None:
            log.warning(
                "profile %r sets no [safety] max_z_jog_um; Z jogs above %s µm "
                "are refused unless forced (assumed default)",
                profile.microscope.name,
                DEFAULT_MAX_Z_JOG_UM,
            )
        self._safety = Safety(
            max_jog_um=safety.max_jog_um,
            max_z_jog_um=safety.max_z_jog_um,
            z_soft_limits_um=safety.z_soft_limits_um,
            xy_soft_limits_um=safety.xy_soft_limits_um,
            max_autofocus_offset_jog=safety.max_autofocus_offset_jog,
            autofocus_offset_soft_limits=safety.autofocus_offset_soft_limits,
            offset_unit=profile.autofocus.offset_unit,
            stop_refused_by=frozenset(role.value for role in safety.stop_refused_by),
            max_engage_excursion_um=safety.max_engage_excursion_um,
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
        #: The channel group, chosen once (``_channel_group``): a stand without
        #: one would otherwise log the same warning at every ``state()``.
        self._channels_group: tuple[str | None] | None = None
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
        core = open_core(
            loaded.config_path(), adapter_search_paths=loaded.adapter_search_dirs()
        )
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

        A profile with a ``[sample]`` table is the demo stand whose camera
        images a synthetic plate: it is installed here, from the profile, so
        ``open``, the CLI, the page and a notebook all get it (m1 §8, "The
        demo-sample stand"). The module that does it is imported inside this
        call, so that no other stand ever loads it.

        Raises:
            ProfileError: The profile has a ``[sample]`` and the stand's camera
                is not the demo's; the facade is closed first.
        """
        microscope = cls(core, profile, dry_run=dry_run)
        if profile.sample is not None:
            from smc.testing.sample_stand import install_sample

            try:
                install_sample(microscope, profile.sample)
            except BaseException:
                # The facade owns the core now, and the caller never gets it to
                # close. A close that fails too must not replace the refusal the
                # operator needs (FM-49): it is logged, and the refusal is raised.
                try:
                    microscope.close()
                except Exception as exc:
                    log.warning(
                        "%s: closing after the failed [sample] install also "
                        "failed (%s)",
                        profile.microscope.name,
                        exc,
                    )
                raise
        return microscope

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
        if found is None and capability is Channels:
            raise CapabilityMissingError(
                "Channels",
                None,
                tuple(
                    str(g)
                    for g in as_hardware_errors(
                        self.core, "channels"
                    ).getAvailableConfigGroups()
                ),
                hint=_NO_CHANNEL_GROUP,
            )
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
        if capability is Autofocus:
            # The interlock trusts only MMAutofocus (_interlock): a Z move on a
            # Nikon stand now leaves PFS off, or fights it.
            log.warning(
                "Autofocus overridden: Z moves no longer suspend and re-engage it"
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
        # The capabilities look their siblings up through the facade, and the
        # facade holds them: a strong reference back would be a cycle that
        # keeps the core alive until the cyclic collector (#142).
        owner = weakref.ref(self)
        if capability is Properties:
            return MMProperties(core, executor)
        if capability is XYStage and (label := get(Role.xy_stage)) is not None:
            return MMXYStage(core, label, executor, safety)
        if capability is ZStage and (label := get(Role.focus)) is not None:
            return MMZStage(
                core,
                label,
                executor,
                safety,
                autofocus=lambda: _owned(owner, "ZStage")._interlock(),
            )
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
        if capability is Autofocus and (label := get(Role.autofocus)) is not None:
            af = self.profile.autofocus
            return MMAutofocus(
                core,
                label,
                executor,
                safety,
                offset_label=get(Role.autofocus_offset),
                offset_unit=af.offset_unit,
                lock_timeout_s=af.lock_timeout_s,
                offset_per_um=af.offset_per_um,
                in_range_property=af.in_range_property,
                in_range_values=af.in_range_values,
                objective_label=self._objective_label(),
                # Lazy: the focus drive is looked up at the call, never here.
                focus=lambda: cast(
                    "ZStage | None", _owned(owner, "Autofocus")._lookup(ZStage)
                ),
                engage_drives_z_to_offset=af.engage_drives_z_to_offset,
                settle_s=af.settle_s,
                settle_tolerance_um=af.settle_tolerance_um,
            )
        if (
            capability is ObjectiveTurret
            and (label := get(Role.objective_turret)) is not None
        ):
            return MMObjectiveTurret(
                core,
                label,
                executor,
                requires_confirm=self.profile.safety.turret_requires_confirm,
                pixel_sizes_um=self.profile.camera.pixel_size_um,
                autofocus=lambda: cast(
                    "Autofocus | None",
                    _owned(owner, "ObjectiveTurret")._lookup(Autofocus),
                ),
            )
        if capability is LightSource and (label := get(Role.light_source)) is not None:
            return MMLightSource(
                core,
                label,
                executor,
                intensity_property=self.profile.light_source.intensity_property,
            )
        if capability is Channels and (group := self._channel_group()) is not None:
            return MMChannels(
                core,
                group,
                executor,
                objective_turret=get(Role.objective_turret),
                exclude=self.profile.channels.exclude,
                config_path=self.profile.config_path(),
                light=lambda: cast(
                    "LightSource | None",
                    _owned(owner, "Channels")._lookup(LightSource),
                ),
            )
        return None

    def _channel_group(self) -> str | None:
        """The config group of ``Channels`` (m2 §8.5), chosen once; ``None``: no group.

        The choice and its warnings are kept: ``state()`` looks ``Channels`` up
        at every call, and a stand with two channel groups would log the same
        warning each time. A read that fails is raised, and the choice is made
        again at the next lookup.
        """
        if self._channels_group is None:
            core = as_hardware_errors(self.core, "channels")
            group, warnings = choose_group(
                [str(g) for g in core.getAvailableConfigGroups()],
                core_group=str(core.getChannelGroup()),
                profile_group=self.profile.channels.group,
            )
            for warning in warnings:
                log.warning("%s", warning)
            self._channels_group = (group,)
        return self._channels_group[0]

    def _interlock(self) -> ZInterlock | None:
        """The autofocus a Z move suspends and engages again; ``None`` when there is none.

        Asked by the Z stage at each move, so neither build calls the other.
        An autofocus that ``override`` replaced is not an ``MMAutofocus`` and
        is not interlocked: a test double cannot be trusted to hold Z. The
        cache lock it takes is re-entrant, since this runs during a Z move,
        where a Ctrl-C handler may land (FM-70).
        """
        found = self._lookup(Autofocus)
        return found if isinstance(found, MMAutofocus) else None

    def _point_autofocus_slot(self) -> None:
        """Point the core's AutoFocus slot at the ``autofocus`` role, once, at construction.

        MMCore's continuous-focus calls take no label: they act on the slot's
        device (FM-87). The role's order of authority (the profile, then the
        slot, then the heuristics, ADR-0003) decides which device that is, so
        a ``[roles.assign]`` or a configuration without ``Core,AutoFocus``
        would otherwise switch one device while the layer reports another.
        A slot that cannot take the device (a role assigned to a device of
        another type) is a WARNING, not a failed open: every autofocus call
        then refuses, naming both, and so does every Z move, which must know
        whether the autofocus is engaged before it moves.
        """
        label = self.roles.get(Role.autofocus)
        if label is None:
            return
        current = self.core.getAutoFocusDevice()
        if current == label:
            return
        try:
            self.core.setAutoFocusDevice(label)
        except Exception as exc:
            # Text, never the error: its traceback holds the core (FM-92).
            log.warning(
                "autofocus: the core's AutoFocus slot names %r and cannot be pointed "
                "at %r (%s); every autofocus call and every Z move will be "
                "refused. Fix the autofocus role under [roles.assign]",
                current,
                label,
                str(exc),
            )
            return
        log.info(
            "autofocus: the core's AutoFocus slot now names %r (was %r), as the "
            "roles say",
            label,
            current,
        )

    def _objective_label(self) -> Callable[[], str | None]:
        """The current objective's name, for the profile's pixel-size map."""
        turret = self.roles.get(Role.objective_turret)
        if turret is None:
            return lambda: None
        core = as_hardware_errors(self.core, Role.objective_turret.value)
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
        autofocus = cast("Autofocus | None", self._lookup(Autofocus))
        if autofocus is not None:
            state.autofocus_engaged = _read("autofocus", autofocus.is_engaged, errors)
            state.autofocus_locked = _read("autofocus", autofocus.is_locked, errors)
            state.autofocus_offset_native = _read(
                "autofocus", autofocus.offset_native, errors
            )
        turret = cast("ObjectiveTurret | None", self._lookup(ObjectiveTurret))
        if turret is not None:
            state.objective = _read("objective", turret.current, errors)
        channels = _read("channel", lambda: self._lookup(Channels), errors)
        if channels is not None:
            state.channel = _read("channel", cast("Channels", channels).current, errors)
        light = cast("LightSource | None", self._lookup(LightSource))
        if light is not None:
            state.light_intensity_native = _read(
                "light", light.intensity_native, errors
            )
        # Read after the devices: "moving" and "halted" are then at least as
        # recent as the positions shown next to them.
        state.moving = self._executor.moving()
        state.halted = self._executor.halted
        return state

    def describe(self) -> str:
        """One status line: ``XY (x, y) µm | Z z µm | exposure ms | shutter open | autofocus off``.

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
        if self._lookup(Autofocus) is not None:
            parts.append(_describe_autofocus(state))
        if state.moving:
            parts.append("moving: " + ", ".join(state.moving))
        if state.halted:
            parts.append("HALTED")
        return " | ".join(parts) or "no stage, camera or shutter"

    # --- stop, resume, close -----------------------------------------------

    def stop(self) -> None:
        """The stand's emergency stop: halt, switch the autofocus off, stop every stage (§13).

        The halt comes first, so no action starts once the stops are under
        way. The autofocus is switched off before the stages, because a Z
        stop sent while PFS still drives Z can be overridden by its next
        correction (m2 §3.11), and its ``autofocus_offset`` device is stopped
        right after it; every one of them is stopped even when an earlier
        one fails. The stops never wait for the microscope lock, so
        they get through a move that holds it. Its log lines come after the
        last stop, ``microscope: halted`` first, then one per stop: a blocked
        console delays the lines, never a stop (FM-62). Every action then
        raises ``MicroscopeHaltedError`` until :meth:`resume`; reads, closing
        a shutter and switching the autofocus off still run.

        A stage the profile declares refuses its stop (``[safety]
        stop_refused_by``) is still sent it; its refusal is no failure, but
        the error says that a move under way runs to its target (m2 §11.5).

        Raises:
            StopRefusedError: Only declared refusals: the message says what the
                stop did and which stages run on; the microscope stays halted.
            HardwareError: One or more stages, the autofocus or its offset
                device did not take the stop; the message names each, and any
                declared refusal, and the microscope stays halted.
        """
        stops = self._executor.hold_log(self._halt_then_stop_stages)
        ending = (
            "The microscope stays halted: stop the stage at the stand if it still "
            "moves, then call `resume()`"
        )
        if stops.failed:
            refused = (
                f"; {join_names(stops.refused)} refused the stop, as the profile "
                f"declares ([safety] stop_refused_by), so a move under way runs to "
                f"its target"
                if stops.refused
                else ""
            )
            raise HardwareError(
                f"the emergency stop failed for {'; '.join(stops.failed)}"
                f"{refused}. {ending}"
            )
        if stops.refused:
            # The autofocus entries of `done`: "autofocus PFS off". The offset
            # device's name starts "autofocus_offset", not "autofocus ".
            off = [done for done in stops.done if done.startswith("autofocus ")]
            switched = f" and switched {join_names(off)}" if off else ""
            raise StopRefusedError(
                tuple(stops.refused),
                f"the emergency stop halted the microscope{switched}, but "
                f"{join_names(stops.refused)} refused the stop, as the profile "
                f"declares ([safety] stop_refused_by): a move under way runs to "
                f"its target. The microscope stays halted: stop the stage with "
                f"the stand's own controls if it must not finish its move, then "
                f"call `resume()`",
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

        A Ctrl-C handler calls :meth:`stop`, not this: landing inside one of
        the ``Executor``'s registry sections, ``close()`` would wait for the
        microscope lock while the interrupted frame holds the registry lock,
        and every other thread's stop would wait with it, up to the lock
        timeout (#76).

        Raises:
            MicroscopeBusyError: Another call held the microscope lock past
                the lock timeout (a hung driver); nothing was unloaded, and
                ``close()`` may be called again.
            HardwareError: ``close()`` was called from inside an action of
                this thread (a Ctrl-C handler that runs during a move, a
                callback inside a snap). Every stage was sent its stop if
                anything moved (the WARNING line says what each stop did),
                but nothing was unloaded: the action still uses the devices.
                Call ``close()`` again once it has returned.
        """
        if self._closed:
            return
        if _owns(self._lock):
            self._stop_if_moving()
            raise HardwareError(
                "close() was called from inside an action on this thread, which "
                "still uses the devices; every stage was sent its stop if anything "
                "moved (the WARNING line says what each stop did), and nothing was "
                "unloaded. Call close() again once the action has returned (after "
                "the `with` block, or in a `finally`)"
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
                # Text, never the error: its traceback holds the facade (FM-92).
                log.error(
                    "%s: unloading the devices failed (%s); the stand may still "
                    "be connected, so restart the program before another one "
                    "connects to it",
                    self.profile.microscope.name,
                    repr(exc),
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
        """Release the stand; a Ctrl-C leaving the block is noted with what is still tracked.

        What the registry tracks is read before ``close()``, which sends the
        stops and may drop a motion that reads idle, and noted in a ``finally``
        after it, so a ``close()`` that raises loses no note. Only for a
        ``KeyboardInterrupt``: it is the one error whose line reads the notes
        (m2 §11.11 E6). Nothing is kept on ``self``, no other lock is taken
        and nothing is written (FM-70).
        """
        tracked = self._executor.moving() if isinstance(exc, KeyboardInterrupt) else ()
        try:
            self.close()
        finally:
            _note_still_moving(exc, tracked)

    def _stop_if_moving(self) -> bool:
        """For ``close()``: stop every stage if anything still moves; whether it did.

        A failed stop is logged, never raised: the devices are unloaded
        anyway, and a stop that did not take is a finding for the log. Every
        stop is sent before any line, the stops' own included (FM-62).

        The line says what each stop did, never "stopped" by default (#206,
        FM-117): a stage that refused its stop runs to its target. Unloading
        does not wait for it: Ctrl-C must let smc exit, and the stand finishes
        the move on its own (m2 §11.5).
        """
        moving = self._executor.moving()
        if not moving:
            return False
        stops = self._executor.hold_log(self._stop_stages)
        outcome: list[str] = []
        if stops.done:
            outcome.append(f"stopped: {', '.join(stops.done)}")
        if stops.refused:
            outcome.append(
                f"refused, as the profile declares ([safety] stop_refused_by), so "
                f"a move under way runs to its target: {', '.join(stops.refused)}"
            )
        if stops.failed:
            outcome.append(f"not stopped: {len(stops.failed)} failed")
        # Logged after the stops were sent (FM-62).
        log.warning(
            "close: %s still moving; %s",
            ", ".join(moving),
            # A stand with no stage and no autofocus has nothing to stop.
            "; ".join(outcome) or "nothing to stop",
        )
        for failure in stops.failed:
            log.error("close: the stop failed for %s", failure)
        return True

    def _halt_then_stop_stages(self) -> _Stops:
        """Halt, then stop every stage: ``stop()`` runs both inside ``hold_log``.

        The halt is inside the hold too, so its line also waits for the last
        stop (§13, FM-62).
        """
        self._executor.halt()
        return self._stop_stages()

    def _stop_stages(self) -> _Stops:
        """Switch the autofocus off, stop its offset device, then each stage; what each did.

        The autofocus first (m2 §3.11): PFS still driving Z could override
        the Z stop with its next correction. Its offset device next: a move
        of it that ``moving()`` reports would otherwise hold the lock until
        its timeout (the review of #140). An interrupt (a second Ctrl-C)
        during one stop does not skip the next: it is re-raised once every
        one has had its stop. The lookup is inside the same guard, since it
        builds a capability nobody used yet. A refusal the profile declares
        is collected apart from the failures (m2 §11.5).
        """
        stops = _Stops()
        interrupted: BaseException | None = None
        for capability, role in _STOPPED:
            try:
                done = self._stop_one(capability, role)
            except StopRefusedError as exc:
                # Once each, in order of first appearance: Z's stop reaches
                # the offset again through its link, and `refused` also names
                # the offset device's own (m2 §11.11 E1).
                for name in exc.refused:
                    if name not in stops.refused:
                        stops.refused.append(name)
            except Exception as exc:
                label = self.roles.get(role)
                kind = role.value if capability is None else capability.__name__
                stops.failed.append(f"{kind}{f' {label!r}' if label else ''} ({exc!r})")
            except BaseException as exc:
                interrupted = interrupted or exc
            else:
                if done is not None:
                    stops.done.append(done)
        try:
            if interrupted is not None:
                raise interrupted
            return stops
        finally:
            # Caught in this frame: exception -> traceback -> this frame -> this
            # local is a cycle that keeps the core until the cyclic collector
            # (#148, FM-92).
            del interrupted

    def _stop_one(self, capability: type | None, role: Role) -> str | None:
        """One entry of ``_STOPPED``: the autofocus switched off, a stage or the offset device stopped.

        The offset device is stopped by its role, whether or not the
        autofocus was built or overridden: its own stop, keyed by its label,
        is the one that cancels a move of it under way. Returns what was done
        (``autofocus PFS off``, or the motion's name), ``None`` when the stand
        has nothing in that role.
        """
        label = self.roles.get(role)
        if capability is None:
            if label is None:
                return None
            motion = autofocus_offset_motion(
                self.core,
                label,
                stop_refused=role.value in self._safety.stop_refused_by,
            )
            self._executor.stop(motion)
            return motion.name
        found = self._lookup(capability)
        if found is None:
            return None
        if capability is Autofocus:
            cast("Autofocus", found).disengage()
            return f"autofocus {label} off" if label else "autofocus off"
        cast("XYStage | ZStage", found).stop()
        name = _STOP_NAMES[role]
        return f"{name} {label}" if label else name


def _owns(lock: threading.RLock) -> bool:
    """Whether the calling thread holds ``lock`` (``RLock._is_owned``, as ``Condition`` uses)."""
    is_owned = getattr(lock, "_is_owned", None)
    return bool(is_owned()) if is_owned is not None else False


def _owned(owner: weakref.ref[Microscope], asked_by: str) -> Microscope:
    """The facade a capability looks its siblings up through; ``HardwareError`` once it is gone.

    ``None`` from those lookups means "this stand has no autofocus" or "no
    focus drive", so a facade that no longer exists must not read as that: a Z
    stage would then move with PFS locked and leave it off (FM-12), and a
    locked ``focus_by_um`` would skip the Z soft-limit check. It raises, and
    the Z stage asks first, before anything is sent. Called under the cache
    lock and from a Ctrl-C handler (FM-70): it takes no lock and does no I/O.

    A Z move that is under way when another thread drops the last reference
    to the facade reaches the relock with the owner gone and raises there,
    PFS off. That is not a supported way to hold a stand.
    """
    microscope = owner()
    if microscope is None:
        raise HardwareError(
            f"{asked_by} belongs to a Microscope that no longer exists, and looks up "
            f"its sibling capabilities (the Z interlock, the focus drive) through "
            f"it; keep the Microscope alive for as long as its capabilities are in use"
        )
    return microscope


def _describe_autofocus(state: MicroscopeState) -> str:
    """The status line's autofocus part; ``autofocus ?`` when a read it needs failed."""
    engaged, locked = state.autofocus_engaged, state.autofocus_locked
    if engaged is None:
        return "autofocus ?"
    if not engaged:
        return "autofocus off"
    if locked is None:
        return "autofocus ?"
    return "autofocus locked" if locked else "autofocus on, not locked"


def _read(name: str, read: Callable[[], R], errors: list[str]) -> R | None:
    """``read()``, or ``None`` with ``"<name>: <error>"`` added to ``errors``."""
    try:
        return read()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        errors.append(f"{name}: {detail}")
        # Text, never the error: a record that keeps it keeps its traceback (FM-92).
        log.debug("state: %s read failed (%s)", name, detail)
        return None
