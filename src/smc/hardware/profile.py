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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from smc.hardware.errors import ProfileError
from smc.hardware.roles import Role

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

__all__ = [
    "CameraSection",
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
    ``.cfg`` can move together.
    """

    config: str = ""
    # A plate traverse exceeds MMCore's 5 s default.
    device_timeout_ms: int = 60_000
    adapter_search_paths: list[str] = Field(default_factory=list)

    @field_validator("device_timeout_ms")
    @classmethod
    def _timeout_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"device_timeout_ms must be a positive number, got {value}"
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


class SafetySection(_Section):
    """Limits the facade enforces; ``None`` means no soft limit on that axis."""

    max_jog_um: float = 5000.0
    z_soft_limits_um: tuple[float, float] | None = None
    xy_soft_limits_um: tuple[tuple[float, float], tuple[float, float]] | None = None
    turret_requires_confirm: bool = True

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
    """Per-objective pixel sizes at binning 1; a missing objective is unknown (0.0)."""

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
            ProfileError: the file is missing, unreadable, not TOML, or does
                not validate. The message names the file, the key and the fix.
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
        path = Path(self.micromanager.config).expanduser()
        if not path.is_absolute() and self.source is not None:
            path = (self.source.parent / path).resolve()
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
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
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


def _load_file(path: Path) -> Profile:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileError(
            f"{path}: cannot read profile ({exc.strerror or exc}). "
            "Check the path, or load it by name from one of the search paths."
        ) from exc
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
    if config is not None and not config.is_file():
        raise ProfileError(
            f"{path}: micromanager.config points to {config}, which does not exist. "
            "Fix the path (relative paths resolve against the profile file; use "
            'forward slashes on Windows too), or set config = "" for the demo '
            "configuration."
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
