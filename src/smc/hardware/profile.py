"""Microscope profiles: one TOML file per instrument (design: docs/design/m1-hardware-layer.md §5).

A profile is the only place where "this stand is special" may be written
(ADR-0003): the Micro-Manager configuration, role overrides, safety limits,
per-objective pixel sizes and quirks. Every section forbids unknown keys, so
a typo in a safety limit is an error at load time instead of a limit that
silently does not apply. ``quirks`` is the one free-form table.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from smc.hardware.errors import ProfileError
from smc.hardware.roles import Role

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

__all__ = [
    "ROLES_WITH_A_STOP",
    "AutofocusSection",
    "CameraSection",
    "ChannelsSection",
    "LightSourceSection",
    "MicroManagerSection",
    "MicroscopeSection",
    "Profile",
    "RolesSection",
    "SafetySection",
    "list_profiles",
    "search_paths",
]

log = logging.getLogger("smc.hardware.profile")

PROFILES_ENV = "SMC_PROFILES"
DEMO_NAME = "demo"
#: MMCore's ``setTimeoutMs`` takes a C ``long``, 32-bit on Windows: a larger
#: timeout passes on Linux and macOS and overflows on the stand's PC. As a lock
#: timeout (``/ 1000``) it also stays under Windows' ``threading.TIMEOUT_MAX``.
_MAX_DEVICE_TIMEOUT_MS = 2**31 - 1
#: The longest lock wait a profile may set: the wait holds the microscope
#: lock, so every other action waits behind it (m2 §3.2). The same bound as
#: ``smc.hardware.backends.mm.MAX_LOCK_TIMEOUT_S``, which this pure module
#: cannot import.
_MAX_LOCK_TIMEOUT_S = 600.0


def _no_boolean(name: str, value: Any) -> Any:
    """Refuse a TOML boolean where a number is expected; anything else passes on.

    pydantic's lax mode reads ``true`` as ``1.0``, so ``lock_timeout_s =
    true`` would load as a one-second wait (#106 fixes the M1 keys; the new
    ones do not add to it). ``bool`` is a subclass of ``int``, hence the
    explicit check, while an integer (``100``) stays a number.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a number, not a boolean (got {str(value).lower()})"
        )
    return value


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MicroscopeSection(_Section):
    """What the instrument is; ``name`` is how it is reported everywhere."""

    name: str
    vendor: str = ""
    description: str = ""


class MicroManagerSection(_Section):
    """How to bring the stand up in Micro-Manager.

    ``config`` empty means the demo configuration shipped with the adapters;
    a relative path resolves against the profile file, so a profile and its
    ``.cfg`` can move together. ``adapter_search_paths`` resolve the same way
    and are searched before pymmcore-plus's own directory (#77).
    """

    config: str = ""
    # A plate traverse exceeds MMCore's 5 s default.
    device_timeout_ms: int = 60_000
    adapter_search_paths: list[str] = Field(default_factory=list)

    @field_validator("adapter_search_paths")
    @classmethod
    def _search_paths_not_empty(cls, value: list[str]) -> list[str]:
        # Path("") is ".": an empty entry would silently mean "the profile's
        # own folder" and put whatever DLLs lie next to the file on the path.
        if any(not entry.strip() for entry in value):
            raise ValueError(
                'adapter_search_paths entries must not be empty; remove the "" entry'
            )
        return value

    @field_validator("device_timeout_ms")
    @classmethod
    def _timeout_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"device_timeout_ms must be a positive number, got {value}"
            )
        if value > _MAX_DEVICE_TIMEOUT_MS:
            raise ValueError(
                f"device_timeout_ms must be at most {_MAX_DEVICE_TIMEOUT_MS} ms "
                f"(about 24.8 days, the largest timeout MMCore accepts on "
                f"Windows), got {value}"
            )
        return value


class RolesSection(_Section):
    """Overrides for role resolution; anything not listed is resolved by heuristics."""

    assign: dict[Role, str] = Field(default_factory=dict)
    exclude: dict[Role, list[str]] = Field(default_factory=dict)


def _check_ordered(limits: tuple[float, float], axis: str) -> None:
    low, high = limits
    # nan comparisons are always False, so "not low < high" already rejects a nan
    # bound — but inf does not ("-1000 < inf" is True), and either way the message
    # below would blame ordering instead of the real problem.
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{axis} limits must be finite numbers, got [{low}, {high}]")
    if not low < high:
        raise ValueError(
            f"{axis} limits must be [low, high] with low < high, got "
            f"[{low}, {high}]; write the lower bound first and make the range non-empty"
        )


#: The roles whose device the layer stops, so the ones ``[safety]
#: stop_refused_by`` may name (m2 §11.3).
ROLES_WITH_A_STOP: tuple[Role, ...] = (Role.xy_stage, Role.focus, Role.autofocus_offset)


class SafetySection(_Section):
    """Limits the facade enforces; ``None`` means no soft limit on that axis.

    ``max_z_jog_um`` is the exception: ``None`` means *not set*, and
    ``Safety`` then assumes ``DEFAULT_MAX_Z_JOG_UM`` and says so. The profile
    keeps ``None`` so that an omitted key stays visible.
    """

    max_jog_um: float = 5000.0
    max_z_jog_um: float | None = None
    z_soft_limits_um: tuple[float, float] | None = None
    xy_soft_limits_um: tuple[tuple[float, float], tuple[float, float]] | None = None
    turret_requires_confirm: bool = True
    #: In ``[autofocus] offset_unit``. ``None``: every relative offset change
    #: needs ``force=True``, since no default exists for a stand's own unit.
    max_autofocus_offset_jog: float | None = None
    #: In ``[autofocus] offset_unit``; not forceable.
    autofocus_offset_soft_limits: tuple[float, float] | None = None
    #: The roles whose device refuses ``stop()``, measured at a stand (m2 §11).
    #: The stop is still sent; the declaration changes what the layer says.
    stop_refused_by: tuple[Role, ...] = ()

    @field_validator("stop_refused_by", mode="before")
    @classmethod
    def _stop_refused_by_roles_with_a_stop(cls, value: Any) -> Any:
        # Before the enum: "camera" is a role, but not one with a stop, and a
        # misspelt role is told the three roles that can be named, not all ten.
        if not isinstance(value, (list, tuple)):
            return value
        names = [getattr(entry, "value", entry) for entry in value]
        allowed = ", ".join(role.value for role in ROLES_WITH_A_STOP)
        for name in names:
            if name not in {role.value for role in ROLES_WITH_A_STOP}:
                raise ValueError(
                    f"{name!r} is not a role with a stop; stop_refused_by lists "
                    f"roles among {allowed}"
                )
        for name in names:
            if names.count(name) > 1:
                raise ValueError(f"stop_refused_by names {name!r} twice; name it once")
        return value

    @field_validator("max_autofocus_offset_jog", mode="before")
    @classmethod
    def _offset_jog_not_boolean(cls, value: Any) -> Any:
        return _no_boolean("max_autofocus_offset_jog", value)

    @field_validator("max_autofocus_offset_jog")
    @classmethod
    def _offset_jog_positive_finite(cls, value: float | None) -> float | None:
        # As for the jog limits in µm: an infinite limit is not "no guard".
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(
                f"max_autofocus_offset_jog must be a positive, finite number in "
                f"[autofocus] offset_unit, got {value}"
            )
        return value

    @field_validator("autofocus_offset_soft_limits", mode="before")
    @classmethod
    def _offset_limits_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            for bound in value:
                _no_boolean("each bound of autofocus_offset_soft_limits", bound)
        return value

    @field_validator("autofocus_offset_soft_limits")
    @classmethod
    def _offset_limits_ordered(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is not None:
            _check_ordered(value, "autofocus offset")
        return value

    @field_validator("max_jog_um")
    @classmethod
    def _max_jog_positive_finite(cls, value: float) -> float:
        # An infinite jog limit is refused, not read as "no jog limit" — turning
        # the jog guard off is not a profile setting.
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"max_jog_um must be a positive, finite number of µm, got {value}"
            )
        return value

    @field_validator("max_z_jog_um")
    @classmethod
    def _max_z_jog_positive_finite(cls, value: float | None) -> float | None:
        # As for max_jog_um: an infinite limit is not "no guard".
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(
                f"max_z_jog_um must be a positive, finite number of µm, got {value}"
            )
        return value

    @field_validator("z_soft_limits_um")
    @classmethod
    def _z_ordered(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        # A reversed pair would make every move look out of range, or none.
        if value is not None:
            _check_ordered(value, "Z")
        return value

    @field_validator("xy_soft_limits_um")
    @classmethod
    def _xy_ordered(
        cls, value: tuple[tuple[float, float], tuple[float, float]] | None
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if value is not None:
            _check_ordered(value[0], "X")
            _check_ordered(value[1], "Y")
        return value


class CameraSection(_Section):
    """Per-objective pixel sizes at binning 1 and a magnification factor of 1.0.

    A missing objective is unknown (0.0). The factor is the core's
    (``getMagnificationFactor()``), which the layer divides by (m2 §8.10).
    """

    pixel_size_um: dict[str, float] = Field(default_factory=dict)

    @field_validator("pixel_size_um")
    @classmethod
    def _positive_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for objective, size in value.items():
            if not math.isfinite(size) or size <= 0:
                raise ValueError(
                    f"pixel size for objective {objective!r} must be a positive "
                    f"number of µm, got {size}"
                )
        return value


class AutofocusSection(_Section):
    """What the stand's continuous autofocus is, as far as it was measured (m2 §3.9).

    Everything defaults to *unknown*: the offset's unit (``""``, shown as
    "device units"), the lock time (``None``: 5 s, assumed and labelled), the
    µm factor per objective (none: ``focus_by_um`` refuses while locked) and
    the in-range property (none: ``is_in_range()`` is ``None``). The values
    come from a session at the stand, never from a guess.
    """

    #: What the offset's numbers are ("PFS offset steps"); "" = unknown.
    offset_unit: str = ""
    #: How long ``engage()`` and the Z interlock wait for a lock; ``None``: 5 s, assumed.
    lock_timeout_s: float | None = None
    #: Objective label -> offset change per µm of focus, measured with the
    #: autofocus locked; sign included.
    offset_per_um: dict[str, float] = Field(default_factory=dict)
    #: A property of the autofocus device that says whether its reference is
    #: in range; "" = the stand does not say.
    in_range_property: str = ""
    #: The values of that property that mean "in range"; any other is out of range.
    in_range_values: list[str] = Field(default_factory=list)

    @field_validator("lock_timeout_s", mode="before")
    @classmethod
    def _lock_timeout_not_boolean(cls, value: Any) -> Any:
        return _no_boolean("lock_timeout_s", value)

    @field_validator("lock_timeout_s")
    @classmethod
    def _lock_timeout_bounded(cls, value: float | None) -> float | None:
        # The wait holds the microscope lock: nan would never end, inf neither.
        if value is not None and not (
            math.isfinite(value) and 0 < value <= _MAX_LOCK_TIMEOUT_S
        ):
            raise ValueError(
                f"lock_timeout_s must be a finite number of seconds in (0, "
                f"{_MAX_LOCK_TIMEOUT_S:g}], got {value}"
            )
        return value

    @field_validator("offset_per_um", mode="before")
    @classmethod
    def _factors_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for objective, factor in value.items():
                _no_boolean(f"offset_per_um for objective {objective!r}", factor)
        return value

    @field_validator("offset_per_um")
    @classmethod
    def _factors_finite_non_zero(cls, value: dict[str, float]) -> dict[str, float]:
        for objective, factor in value.items():
            # A zero factor would turn every focus_by_um into no move at all,
            # reported as done; nan or inf would reach the offset device.
            if not math.isfinite(factor) or factor == 0:
                raise ValueError(
                    f"offset_per_um for objective {objective!r} must be a finite, "
                    f"non-zero number (the offset change per µm of focus, measured "
                    f"with the autofocus locked), got {factor}"
                )
        return value

    @model_validator(mode="after")
    def _in_range_together(self) -> AutofocusSection:
        # One without the other cannot say "in range": a property with no
        # values would read every value as out of range, and the reverse
        # names nothing to read.
        has_property = bool(self.in_range_property.strip())
        if self.in_range_property and not has_property:
            raise ValueError(
                "in_range_property must name a property of the autofocus device, "
                f"or be empty; got {self.in_range_property!r}"
            )
        if has_property != bool(self.in_range_values):
            raise ValueError(
                "in_range_property and in_range_values go together: set both (the "
                "property and the values of it that mean in range), or neither"
            )
        if any(not entry.strip() for entry in self.in_range_values):
            raise ValueError(
                'in_range_values must not contain an empty value; remove the "" entry'
            )
        return self


class LightSourceSection(_Section):
    """How bright the ``light_source`` device is set (m2 §8.4)."""

    #: The light source's property that sets how bright; "" = found by hints.
    intensity_property: str = ""


class ChannelsSection(_Section):
    """Which config group holds the channels, and what a capture leaves out (m2 §8.5)."""

    #: The config group whose presets are channels; "" = chosen by its name.
    group: str = ""
    #: Device labels that ``capture()`` never records.
    exclude: list[str] = Field(default_factory=list)

    @field_validator("group")
    @classmethod
    def _group_fits_a_cfg_line(cls, value: str) -> str:
        # Written into the .cfg as a field of a comma-separated line (FM-81).
        if value and (
            "," in value
            or value != value.strip()
            or not all(" " <= c <= "~" for c in value)
        ):
            raise ValueError(
                f"group {value!r} cannot be written to a configuration file: use "
                "printable ASCII with no comma and no leading or trailing space"
            )
        return value

    @field_validator("exclude")
    @classmethod
    def _exclude_not_empty(cls, value: list[str]) -> list[str]:
        if any(not entry.strip() for entry in value):
            raise ValueError(
                'exclude entries must name a device label; remove the "" entry'
            )
        return value


class Profile(_Section):
    """One instrument's stand-specific facts, validated.

    ``source`` is where the profile was loaded from (``None`` for
    :meth:`demo`); it is set by :meth:`load` and never serialised, so a
    dumped profile is independent of where the file lived.
    """

    microscope: MicroscopeSection
    micromanager: MicroManagerSection = Field(default_factory=MicroManagerSection)
    roles: RolesSection = Field(default_factory=RolesSection)
    safety: SafetySection = Field(default_factory=SafetySection)
    camera: CameraSection = Field(default_factory=CameraSection)
    autofocus: AutofocusSection = Field(default_factory=AutofocusSection)
    light_source: LightSourceSection = Field(default_factory=LightSourceSection)
    channels: ChannelsSection = Field(default_factory=ChannelsSection)
    quirks: dict[str, Any] = Field(default_factory=dict)
    source: Path | None = Field(default=None, exclude=True)

    @classmethod
    def demo(cls) -> Profile:
        """The simulator profile, built in code so it works with no file at all.

        It mirrors ``profiles/demo.toml`` exactly; a test enforces that.
        """
        return cls(
            microscope=MicroscopeSection(
                name="demo",
                vendor="Micro-Manager",
                description=(
                    "DemoCamera adapter: simulated camera, XY stage, Z, turret, "
                    "shutter, autofocus."
                ),
            ),
            micromanager=MicroManagerSection(
                config="", device_timeout_ms=60_000, adapter_search_paths=[]
            ),
            safety=SafetySection(
                max_jog_um=5000.0,
                max_z_jog_um=100.0,
                z_soft_limits_um=(-1000.0, 1000.0),
                turret_requires_confirm=True,
            ),
            camera=CameraSection(
                pixel_size_um={
                    "Nikon 10X S Fluor": 0.65,
                    "Nikon 40X Plan Fluor ELWD": 0.1625,
                }
            ),
        )

    @classmethod
    def load(cls, source: str | Path) -> Profile:
        """Load a profile from a path, or by name over :func:`search_paths`.

        A :class:`~pathlib.Path`, or a string that contains a directory
        separator or ends in ``.toml``, is read as given. Anything else is a
        name resolved to ``<dir>/<name>.toml``; ``"demo"`` falls back to
        :meth:`demo` when no ``demo.toml`` is found.

        Raises:
            ProfileError: the file is missing, unreadable, not UTF-8 (a
                UTF-8 byte-order mark is accepted), not TOML, or does not
                validate. The message names the file, the key or line and the
                fix.
        """
        if _is_path(source):
            return _load_file(Path(source))
        name = str(source)
        found = _find(name)
        if found is not None:
            return _load_file(found)
        if name == DEMO_NAME:
            log.debug("no demo.toml on the search path; using Profile.demo()")
            return cls.demo()
        raise ProfileError(_not_found_message(name))

    def config_path(self) -> Path | None:
        """The Micro-Manager configuration file, or ``None`` for the demo configuration.

        A relative ``config`` resolves against the profile file's directory
        (against the working directory for a profile built in code).
        """
        if not self.micromanager.config:
            return None
        return self._resolve_entry("config", self.micromanager.config)

    def adapter_search_dirs(self) -> list[Path]:
        """The extra adapter directories, in file order, resolved like :meth:`config_path`.

        Order is the priority (the first directory wins when two hold the
        same adapter), so it is kept, and nothing is de-duplicated here:
        ``open_core`` does that once it knows what the core already searches.
        The Zeiss PC's MMStudio folder goes here, for an adapter that only
        the GUI install has (#77).
        """
        return [
            self._resolve_entry("adapter_search_paths", e)
            for e in self.micromanager.adapter_search_paths
        ]

    def _resolve_entry(self, key: str, entry: str) -> Path:
        # One rule for every path in the profile: ``~`` expands, and a
        # relative path follows the file, not the working directory.
        try:
            path = Path(entry).expanduser()
        except RuntimeError as exc:
            # "~user" with no such user, or no home directory: a ProfileError,
            # so that a listing of profiles reports this one and goes on.
            where = f"{self.source}: " if self.source is not None else ""
            raise ProfileError(
                f"{where}micromanager.{key}: cannot expand '~' in {entry} ({exc}). "
                "Write an absolute path, or set HOME (USERPROFILE on Windows)."
            ) from exc
        if not path.is_absolute() and self.source is not None:
            try:
                path = (self.source.parent / path).resolve()
            except (OSError, ValueError) as exc:
                # An embedded NUL, or a drive that does not answer: a
                # ProfileError, so that a listing reports this profile and goes on.
                raise ProfileError(
                    f"{self.source}: micromanager.{key}: cannot resolve {entry} "
                    f"({getattr(exc, 'strerror', None) or exc}). Write a path the "
                    "operating system accepts."
                ) from exc
        return path


def search_paths() -> list[Path]:
    """Directories searched for ``<name>.toml``, in priority order.

    ``$SMC_PROFILES`` (``os.pathsep``-separated, so ``;`` on Windows) comes
    first so a site can keep its profiles outside the repository, then
    ``./profiles``, then the working directory.
    """
    env = os.environ.get(PROFILES_ENV, "")
    dirs = [Path(entry).expanduser() for entry in env.split(os.pathsep) if entry]
    cwd = Path.cwd()
    return [*dirs, cwd / "profiles", cwd]


def list_profiles() -> list[tuple[str, Path]]:
    """Every ``(name, path)`` a bare name would resolve to, sorted by name.

    When the same name exists in several search paths only the one
    :meth:`Profile.load` would pick (the first) is listed.
    """
    found: dict[str, Path] = {}
    for directory in search_paths():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            if path.stem not in found and _may_be_profile(path):
                found[path.stem] = path.resolve()
    return sorted(found.items())


def _may_be_profile(path: Path) -> bool:
    """False only for TOML that parses and has no ``[microscope]`` table.

    The working directory is a search path, so ``pyproject.toml`` would
    otherwise be listed as a profile. A file that cannot be read or parsed
    stays listed: loading it is how its owner learns what is wrong.
    """
    try:
        data = tomllib.loads(_read_text(path))
    except (ProfileError, tomllib.TOMLDecodeError):
        return True
    return "microscope" in data


# -- loading ------------------------------------------------------------------


def _is_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    return source.endswith(".toml") or "/" in source or os.sep in source


def _find(name: str) -> Path | None:
    for directory in search_paths():
        candidate = directory / f"{name}.toml"
        if candidate.is_file():
            return candidate
    return None


def _not_found_message(name: str) -> str:
    looked = "\n".join(f"  - {d}" for d in search_paths())
    available = ", ".join(n for n, _ in list_profiles()) or "none"
    return (
        f"No profile named {name!r}; looked for {name}.toml in:\n{looked}\n"
        f"Available profiles: {available}. Pass a path to the .toml file, or add "
        f"its directory to {PROFILES_ENV} (separated by {os.pathsep!r})."
    )


def _read_text(path: Path) -> str:
    """The profile's text, or a ``ProfileError`` that says what to do about the file.

    Read as bytes and decoded as ``utf-8-sig``: Notepad before Windows 10 1903
    starts a "UTF-8" file with a byte-order mark, which ``tomllib`` refuses
    with an error at a line that is fine, and it also offers cp1252 and
    UTF-16, which no decoder here can tell from a typo (FM-80).
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProfileError(
            f"{path}: cannot read profile ({exc.strerror or exc}). "
            "Check the path, or load it by name from one of the search paths."
        ) from exc
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        # ``exc.start`` indexes ``exc.object``, which for utf-8-sig is the data
        # after the mark: ``raw`` would name a byte three places off.
        seen: bytes = exc.object
        line = seen.count(b"\n", 0, exc.start) + 1
        raise ProfileError(
            f"{path}: line {line} is not UTF-8 text ({exc.reason}, byte "
            f"0x{seen[exc.start]:02x}). Save the file as UTF-8 "
            "(Notepad: File > Save As > Encoding: UTF-8)."
        ) from exc


def _exists(source: Path, key: str, target: Path, *, directory: bool) -> bool:
    """Whether ``target`` is a file (a directory); a ``ProfileError`` if the OS will not say.

    ``Path.is_file`` answers False for a path that is not there but re-raises
    the rest: a folder this account may not search, a network share that
    needs a login. A listing of profiles reports the profile and goes on, so
    the loader raises only ``ProfileError`` for a problem with the file (FM-77).
    """
    try:
        return target.is_dir() if directory else target.is_file()
    except OSError as exc:
        raise ProfileError(
            f"{source}: micromanager.{key}: cannot look at {target} "
            f"({exc.strerror or exc}). Check that this account may read it (a "
            "network share may need a login)."
        ) from exc


def _load_file(path: Path) -> Profile:
    text = _read_text(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"{path}: not valid TOML: {exc}") from exc

    if "source" in data:
        raise ProfileError(
            f"{path}: key 'source' is set by the loader, not the file — remove it."
        )
    try:
        profile = Profile.model_validate(data)
    except ValidationError as exc:
        raise ProfileError(_validation_message(path, exc)) from exc

    profile.source = path.resolve()
    config = profile.config_path()
    if config is not None and not _exists(path, "config", config, directory=False):
        raise ProfileError(
            f"{path}: micromanager.config points to {config}, which does not exist. "
            "Fix the path (relative paths resolve against the profile file; use "
            'forward slashes on Windows too), or set config = "" for the demo '
            "configuration."
        )
    # MMCore accepts a directory that does not exist without a word, so a
    # typo here would load nothing from it and say nothing (#77).
    for entry, directory in zip(
        profile.micromanager.adapter_search_paths,
        profile.adapter_search_dirs(),
        strict=True,
    ):
        if not _exists(path, "adapter_search_paths", directory, directory=True):
            looked = "" if str(directory) == entry else f" (looked for {directory})"
            raise ProfileError(
                f"{path}: micromanager.adapter_search_paths lists {entry}, which is "
                f"not a directory{looked}. Fix the path (relative paths resolve "
                "against the profile file; use forward slashes on Windows too), or "
                "remove the entry."
            )
    log.debug("loaded profile %r from %s", profile.microscope.name, path)
    return profile


# The model behind each table, to list the keys that do exist.
_SECTIONS: dict[tuple[str, ...], type[BaseModel]] = {
    (): Profile,
    ("microscope",): MicroscopeSection,
    ("micromanager",): MicroManagerSection,
    ("roles",): RolesSection,
    ("safety",): SafetySection,
    ("camera",): CameraSection,
    ("autofocus",): AutofocusSection,
    ("light_source",): LightSourceSection,
    ("channels",): ChannelsSection,
}


def _allowed_keys(section: tuple[str, ...]) -> str:
    model = _SECTIONS.get(section)
    if model is None:
        return ""
    keys = [k for k, f in model.model_fields.items() if not f.exclude]
    return ", ".join(keys)


def _validation_message(path: Path, exc: ValidationError) -> str:
    lines = [f"{path}: invalid profile ({exc.error_count()} problem(s))"]
    for error in exc.errors():
        loc = tuple(str(part) for part in error["loc"])
        is_key = bool(loc) and loc[-1] == "[key]"
        parts = loc[:-1] if is_key else loc
        key = ".".join(parts)
        kind = error["type"]
        if kind == "extra_forbidden":
            table = ".".join(parts[:-1])
            allowed = _allowed_keys(parts[:-1])
            problem = f"unknown key in [{table}]" if table else "unknown section"
            fix = f"fix the spelling or remove it; allowed keys: {allowed}"
        elif kind == "enum" and is_key:
            problem = "not a role"
            fix = "use one of: " + ", ".join(r.value for r in Role)
        elif kind == "missing":
            problem = "required but missing"
            fix = f"add [{key}]" if len(parts) == 1 else f"add {parts[-1]} = ..."
        elif kind == "value_error":
            # Our validators write the problem and the fix in one sentence.
            lines.append(f"  - {key}: {error['msg'].removeprefix('Value error, ')}")
            continue
        else:
            problem = error["msg"]
            fix = f"got {error['input']!r}"
        lines.append(f"  - {key}: {problem} — {fix}")
    return "\n".join(lines)
