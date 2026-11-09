"""Presets: the choices an assay repeats, named and kept in a TOML file (m5 §10).

A preset holds the plate type, the wells and the plugins' ``--param`` pairs.
It is a convenience and never the truth: the run document keeps the values a
run used, and only names the preset it was given. A preset holds no position
(``refs``, ``walls``) and moves nothing.

Two places read and write presets, the page and the command line, so the file
and its check live here and import no Panel and reach no device:

* **The file** is TOML, written by ``tomli-w`` and read as ``utf-8-sig`` (a
  byte-order mark from Notepad is not an error, FM-80), saved through
  ``write_atomically`` so that a Ctrl-C or a refused rename keeps the old
  preset (FM-38).
* **The name** is the file's stem and is lower case and short, so that two
  names never differ by case alone on NTFS and APFS, and no name is one that
  Windows keeps for a device (FM-24).
* **The check** says what of a preset a stand can use. A mismatch is a note,
  never a refusal: a plugin that renamed one parameter keeps the rest of the
  assay (§10.2), and the owner reads what was not applied.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import useq
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_core import to_jsonable_python

from smc.hardware.profile import Profile, check_pairs, search_paths
from smc.plugins.api import PluginParams
from smc.plugins.errors import ParamsError
from smc.plugins.params import format_param, parse_params
from smc.plugins.scan.algorithm import RESERVED_NAMES
from smc.run.document import RunDocument, StageStatus
from smc.run.files import AtomicWriteError, retry_on_permission_error, write_atomically

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

__all__ = [
    "NOT_IN_PARAMS",
    "Preset",
    "PresetCheck",
    "PresetError",
    "PresetExistsError",
    "PresetFile",
    "PresetFromRun",
    "check_name",
    "check_preset",
    "command_pairs",
    "dumps_preset",
    "list_presets",
    "load_preset",
    "preset_dirs",
    "preset_from_document",
    "registered_plate",
    "save_preset",
]

log = logging.getLogger("smc.plugins.presets")

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_HEADER = (
    "# A preset of smc: the choices an assay repeats. "
    "See docs/ui/plate-page.md, Presets.\n"
)

#: What a plugin's ``Params`` has as a field but a preset keeps elsewhere: the
#: plate and the wells are the preset's own fields, and positions are never
#: kept (m5 §10.2). The turn confirmation of ``objective-offsets`` is not kept
#: either: it says the stage is clear *now*, so a saved one would confirm the next
#: session's turns over a loaded plate for the operator (FM-14). Per plugin, the
#: key and the reason the note gives.
NOT_IN_PARAMS: dict[str, dict[str, str]] = {
    "objective-offsets": {
        "confirm_turns": "the stage is clear only for the run that confirmed it: "
        "every run confirms it again"
    },
    "plate-finding": {
        "centre_x_um": "a preset holds no positions",
        "centre_y_um": "a preset holds no positions",
    },
    "plate-registration": {
        "plate": "the preset's plate holds it",
        "refs": "a preset holds no positions",
        "walls": "a preset holds no positions",
        "objective": "the registration takes the objective in place",
    },
    "plate-walls": {
        "plate": "the preset's plate holds it",
        "refs": "a preset holds no positions",
    },
    "scan-plan": {"wells": "the preset's wells hold them"},
}

#: The plugin whose parameter a preset's own field stands for: the parameter's
#: key, and the ``PresetCheck`` attribute that holds its value. A plugin whose
#: ``plate`` is in ``NOT_IN_PARAMS`` is here too, or a run from the preset would
#: use the plugin's default plate where the preset named one (#371).
_HELD_FIELDS: dict[str, tuple[str, str]] = {
    "plate-registration": ("plate", "plate"),
    "plate-walls": ("plate", "plate"),
    "scan-plan": ("wells", "wells"),
}


class PresetError(Exception):
    """A preset cannot be read, named or saved; the message is the whole diagnosis, on one line."""


class PresetExistsError(PresetError):
    """``save_preset`` found a preset of that name; ``path`` is the file that holds it."""

    def __init__(self, name: str, path: Path) -> None:
        super().__init__(f"preset {name} exists: {path}")
        self.name = name
        self.path = path

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuilt from the arguments, not from ``args``, as the errors of
        # ``smc.hardware`` are, so that ``path`` survives a worker process.
        return (type(self), (self.name, self.path))


class Preset(BaseModel):
    """What a preset file holds; every field may be left out (m5 §10.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The profile's ``microscope.name`` where it was saved.
    stand: str | None = None
    #: A ``useq`` plate key.
    plate: str | None = None
    #: The text ``scan-plan``'s ``wells`` takes (``A1:D6``, ``C*``, ``B2,B11``).
    wells: str | None = None
    saved_utc: AwareDatetime | None = None
    #: Per plugin name, ``smc run --param`` pairs, as ``[suggest]`` has them:
    #: the preset *is* the command's call.
    params: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("stand", "plate", "wells")
    @classmethod
    def _one_non_blank_line(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is not None and (not value.strip() or "\n" in value or "\r" in value):
            raise ValueError(
                f"{info.field_name} is one non-empty line; leave it out for none"
            )
        return value

    @field_validator("params")
    @classmethod
    def _each_list_is_pairs(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        # The same check and the same texts as ``[suggest]``'s; an empty list
        # is allowed here, where a suggestion needs one item.
        for plugin, items in value.items():
            try:
                check_pairs(items)
            except ValueError as exc:
                raise ValueError(f"{plugin}: {exc}") from None
        return value


@dataclass(frozen=True, slots=True)
class PresetFile:
    """A preset and where it came from, or why it could not be read (``preset`` is ``None``)."""

    name: str
    path: Path
    preset: Preset | None
    error: str | None = None


# -- the name and the folders ---------------------------------------------------


def check_name(name: str) -> str:
    """``name`` when it can be a preset's file stem, else a ``PresetError``.

    Lower case only, so that two presets never differ by case alone (NTFS and
    the default APFS would read them as one file, FM-24), and not a name that
    Windows keeps for a device (``nul``, ``com1``): ``nul.toml`` is the null
    device there.
    """
    if _NAME.fullmatch(name) is None:
        raise PresetError(
            "a preset's name is 1 to 64 lower-case letters, digits, - and _, "
            f"starting with a letter or a digit; got '{name}'"
        )
    if name.upper() in RESERVED_NAMES:
        raise PresetError(
            f"'{name}' is a name Windows keeps for a device; choose another"
        )
    return name


def preset_dirs(profile: Profile) -> tuple[Path, ...]:
    """The folders that hold presets, in the order a name is looked for (m5 §10.1).

    The profile's own ``presets/`` first, when it was loaded from a file, so that
    a preset of the stand sits beside its profile; then ``presets/`` of every
    profile search path, which serves every stand of a checkout. A folder that
    two spellings name once is kept once: ``os.path.normcase`` makes the
    comparison fold case where the file system does (FM-24).
    """
    candidates: list[Path] = []
    if profile.source is not None:
        candidates.append(profile.source.parent / "presets")
    candidates.extend(directory / "presets" for directory in search_paths())
    seen: set[str] = set()
    folders: list[Path] = []
    for folder in candidates:
        key = os.path.normcase(os.path.abspath(folder))
        if key not in seen:
            seen.add(key)
            folders.append(folder)
    return tuple(folders)


def _is_path(source: str) -> bool:
    """A path is what ``Profile.load`` takes for one: ``.toml``, or a separator in it."""
    return source.endswith(".toml") or "/" in source or os.sep in source


# -- reading --------------------------------------------------------------------


def _read(path: Path) -> Preset:
    """The preset in ``path``; a ``PresetError`` whose message is the diagnosis."""
    try:
        raw = retry_on_permission_error(path.read_bytes)
    except OSError as exc:
        raise PresetError(f"{path}: cannot be read: {exc.strerror or exc}") from exc
    try:
        # utf-8-sig: the byte-order mark Notepad writes is not an error (FM-80).
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PresetError(
            f"{path}: cannot be read: not UTF-8 text ({exc.reason} at byte "
            f"{exc.start}); save it as UTF-8"
        ) from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PresetError(f"{path}: not TOML: {' '.join(str(exc).split())}") from exc
    try:
        return Preset.model_validate(data)
    except ValidationError as exc:
        problems = [_problem(error) for error in exc.errors()]
        raise PresetError(f"{path}: {'; '.join(problems)}") from None


def _problem(error: Any) -> str:
    where = ".".join(str(part) for part in error["loc"])
    if error["type"] == "extra_forbidden":
        fields = ", ".join(name for name in Preset.model_fields)
        message = f"unknown key; the keys are {fields}"
    else:
        message = " ".join(str(error["msg"]).removeprefix("Value error, ").split())
    return f"{where}: {message}" if where else message


def list_presets(profile: Profile) -> tuple[PresetFile, ...]:
    """Every ``*.toml`` of every preset folder, sorted by name; a broken one is kept with its error.

    The first file of a name wins, as for profiles. A file that cannot be read,
    is not a valid preset or whose stem ``check_name`` refuses is listed with
    ``preset=None`` and its ``error``, never dropped: a failure is a finding.
    A folder that cannot be listed is logged and skipped.
    """
    found: dict[str, PresetFile] = {}
    for folder in preset_dirs(profile):
        try:
            if not folder.is_dir():
                continue
            paths = sorted(folder.glob("*.toml"))
        except OSError as exc:
            log.warning("presets: cannot list %s: %s", folder, exc.strerror or exc)
            continue
        for path in paths:
            name = path.stem
            if name in found:
                continue
            try:
                check_name(name)
                found[name] = PresetFile(name, path, _read(path))
            except PresetError as exc:
                found[name] = PresetFile(name, path, None, str(exc))
    return tuple(found[name] for name in sorted(found))


def load_preset(source: str, profile: Profile) -> PresetFile:
    """The preset ``source`` names: a path as given, else a name looked for in ``preset_dirs``.

    A path is read as the profile's rule reads one (it ends in ``.toml`` or has
    a separator), and its name is its stem. Any other source is a name: it goes
    through ``check_name`` and is looked for as ``<dir>/<name>.toml`` in the
    folders in order.

    Raises:
        PresetError: No such preset (the message lists the folders searched),
            the file cannot be read, is not TOML, or is not a valid preset.
            The message is the whole diagnosis, on one line.
    """
    if _is_path(source):
        given = Path(source)
        return PresetFile(given.stem, given, _read(given))
    name = check_name(source)
    path = _find(name, profile)
    if path is None:
        looked = ", ".join(str(folder) for folder in preset_dirs(profile))
        raise PresetError(
            f"no preset named {name}; looked in {looked}; smc presets list shows "
            "the presets"
        )
    return PresetFile(name, path, _read(path))


def _find(name: str, profile: Profile) -> Path | None:
    for folder in preset_dirs(profile):
        candidate = folder / f"{name}.toml"
        try:
            if candidate.is_file():
                return candidate
        except OSError as exc:
            raise PresetError(
                f"{candidate}: cannot be looked at: {exc.strerror or exc}"
            ) from exc
    return None


# -- writing --------------------------------------------------------------------


def dumps_preset(preset: Preset) -> str:
    """The text of the file: a header comment, then the preset as TOML.

    Fields that are ``None`` are left out, so a preset that names no wells has
    no ``wells`` key. ``tomli-w`` quotes a key only where TOML needs it
    (``"my.plugin"``) and writes the time as ``2026-11-03 10:00:00+00:00``.

    ``tomli-w`` is imported here and not at the top: only a save needs it, and
    this module is imported by every ``smc`` command. A PC updated with ``git
    pull`` and not reinstalled (no internet at the stand) must still run
    ``smc doctor`` and the rest, and say what is missing when a preset is saved.

    Raises:
        PresetError: ``tomli-w`` is not installed.
    """
    try:
        import tomli_w
    except ImportError:
        raise PresetError(
            "saving a preset needs the tomli-w package, which this install lacks: "
            "run pip install -e . in the smc folder, then save again"
        ) from None
    return _HEADER + tomli_w.dumps(preset.model_dump(mode="python", exclude_none=True))


def save_preset(
    name: str, preset: Preset, profile: Profile, *, overwrite: bool = False
) -> Path:
    """Write ``preset`` as ``name`` into the first preset folder and return the file.

    The write is atomic (FM-38): the old preset or the whole new one, and when
    only the rename is refused (Windows, a program holding the file) the new
    text is kept beside it and the error names it. A preset found in a later
    folder is shadowed by the file written in the first.

    Raises:
        PresetError: The name is refused, or the file cannot be written.
        PresetExistsError: A preset of that name exists, in any folder, and
            ``overwrite`` is false; ``path`` is the file that exists.
    """
    check_name(name)
    existing = _find(name, profile)
    if existing is not None and not overwrite:
        raise PresetExistsError(name, existing)
    path = preset_dirs(profile)[0] / f"{name}.toml"
    try:
        write_atomically(path, dumps_preset(preset).encode("utf-8"))
    except AtomicWriteError as exc:
        kept = f"; the new preset is in {exc.kept}" if exc.kept is not None else ""
        raise PresetError(
            f"{path}: not saved: {exc.strerror or exc.cause}{kept}"
        ) from exc
    return path


# -- the check ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PresetCheck:
    """What of a preset applies on a stand, and what does not and why.

    ``plate`` and ``wells`` are the preset's when applied, else ``None``.
    ``params`` keeps the applied pairs in the file's order; a plugin left with
    none is not a key.
    """

    name: str
    plate: str | None
    wells: str | None
    params: dict[str, tuple[str, ...]]
    mismatches: tuple[str, ...]


def registered_plate(doc: RunDocument) -> str | None:
    """The plate this run document has registered: the latest ``done`` record of the ``plate`` slot.

    ``plate-registration`` and ``plate-finding`` both provide it; the one that
    finished last is the plate the next stages use. ``None`` when there is none,
    or when its result holds no plate name. It reads the document only.
    """
    filling = [
        record
        for record in doc.stages.values()
        if record.provides == "plate" and record.status is StageStatus.done
    ]
    if not filling:
        return None
    latest = max(filling, key=lambda r: (r.finished_utc or r.started_utc, r.plugin))
    plate = (latest.result or {}).get("plate")
    return plate if isinstance(plate, str) else None


def check_preset(
    file: PresetFile,
    models: Mapping[str, type[BaseModel]],
    *,
    stand: str,
    registered: str | None,
    only: str | None = None,
) -> PresetCheck:
    """What of ``file``'s preset applies on ``stand``, with one line for everything that does not (§10.2).

    ``models`` maps a plugin's name to its ``Params``. With ``only``, only that
    plugin's pairs are checked and returned, and the other plugins of the file
    are not mismatches: the command runs one plugin and imports no other. A bad
    key drops only its own pair, where a bad ``[suggest]`` is ignored whole: a
    plugin that renamed one parameter keeps the rest of the assay. A value is
    not checked; a value its plugin refuses is that run's ``ParamsError``.

    Raises:
        PresetError: ``file`` holds no preset (it could not be read).
    """
    preset = file.preset
    if preset is None:
        raise PresetError(f"{file.path}: {file.error or 'holds no preset'}")
    n = file.name
    mismatches: list[str] = []

    if preset.stand is not None and preset.stand != stand:
        mismatches.append(
            f"preset {n} was saved on {preset.stand}; this stand is {stand}"
        )

    plate, wells = preset.plate, preset.wells
    if plate is not None and plate not in useq.registered_well_plate_keys():
        mismatches.append(f"preset {n}: {plate} is not a plate useq knows")
        plate = wells = None
    elif plate is not None and registered is not None and registered != plate:
        mismatches.append(
            f"preset {n} is for a {plate} plate; this run document's plate is "
            f"registered as a {registered}: its plate and wells are not applied"
        )
        plate = wells = None

    params: dict[str, tuple[str, ...]] = {}
    for plugin, items in preset.params.items():
        if only is not None and plugin != only:
            continue
        model = models.get(plugin)
        if model is None:
            mismatches.append(f"preset {n} [params.{plugin}]: no plugin named {plugin}")
            continue
        kept: list[str] = []
        for item in items:
            key = item.partition("=")[0]
            if key in NOT_IN_PARAMS.get(plugin, {}):
                mismatches.append(
                    f"preset {n} [params.{plugin}]: {key} is not kept under "
                    f"params: {NOT_IN_PARAMS[plugin][key]}"
                )
            elif key not in model.model_fields:
                fields = ", ".join(model.model_fields)
                mismatches.append(
                    f"preset {n} [params.{plugin}]: {key} is not a parameter of "
                    f"{plugin} (keys: {fields})"
                )
            else:
                kept.append(item)
        if kept:
            params[plugin] = tuple(kept)
    return PresetCheck(n, plate, wells, params, tuple(mismatches))


def command_pairs(
    check: PresetCheck, plugin: str, overrides: Sequence[str]
) -> list[str]:
    """The ``--param`` pairs ``smc run --preset`` runs ``plugin`` with, in order (§10.2).

    First the preset's own field when ``plugin`` takes it (``plate`` for
    ``plate-registration``, ``wells`` for ``scan-plan``) and it was applied;
    then the preset's pairs for ``plugin``; then ``overrides``. A key that an
    override gives is dropped from the first two, so ``--param`` wins. The key
    of an item is its text before the first ``=``. A malformed override is
    passed on as it is: ``parse_params`` names it.
    """
    taken = {item.partition("=")[0] for item in overrides}
    pairs: list[str] = []
    held = _HELD_FIELDS.get(plugin)
    if held is not None:
        key, attribute = held
        value = getattr(check, attribute)
        if value is not None:
            pairs.append(f"{key}={value}")
    pairs.extend(check.params.get(plugin, ()))
    kept = [item for item in pairs if item.partition("=")[0] not in taken]
    return [*kept, *overrides]


# -- a preset from a run document -----------------------------------------------


@dataclass(frozen=True, slots=True)
class PresetFromRun:
    """A preset made from a run document, and one note for everything left out of it."""

    preset: Preset
    notes: tuple[str, ...]


def preset_from_document(
    doc: RunDocument,
    models: Mapping[str, type[PluginParams]],
    *,
    stand: str,
    saved_utc: datetime,
) -> PresetFromRun:
    """The preset the ``done`` records of ``doc`` would be run again from (m5 §10.9).

    Only a finished run's values are the assay's: a ``stopped``, ``failed`` or
    ``running`` record is left out, and a dry run's is in, its values being the
    operator's. A pair that equals its plugin's default is left out too, so the
    preset follows the plugin's default as ``[suggest]`` does. What cannot be
    saved is a note, never silence (FM-107): a plugin whose pairs would be read
    back as other values loses all of them, because a subset may be refused
    (``rows`` without ``columns``), and the note names the key.

    Raises:
        PresetError: The document has no finished stage, saves nothing, or
            holds a ``wells`` text that is not one line.
    """
    done = {
        plugin: record
        for plugin, record in sorted(doc.stages.items())
        if record.status is StageStatus.done
    }
    if not done:
        raise PresetError(
            "the run document has no finished stage; there is nothing to save"
        )
    params: dict[str, tuple[str, ...]] = {}
    notes: list[str] = []
    only_defaults: list[str] = []
    wells: str | None = None
    for plugin, record in done.items():
        pairs, note = _pairs_of(plugin, models.get(plugin), record.params)
        if pairs:
            params[plugin] = pairs
        elif note is None:
            only_defaults.append(plugin)
        if note is not None:
            notes.append(note)
        if plugin == "scan-plan":
            wells = record.params.get("wells")
            if not isinstance(wells, str) or not wells.strip():
                wells = None
            elif not _is_text(wells):
                notes.append(
                    "scan-plan: wells holds a character that is not text; not saved"
                )
                wells = None
        if record.profile != stand:
            notes.append(f"{plugin} was run on {record.profile}, not on {stand}")
    plate = registered_plate(doc)
    if plate is not None and not _is_text(plate):
        # A plate name read from a result that carries a lone surrogate (FM-05):
        # it would end the save at the encoding, with no word about which value.
        notes.append("plate holds a character that is not text; not saved")
        plate = None
    if plate is None and wells is None and not params:
        # The refused plugins are named too: "used only default values" would be
        # untrue of a plugin whose values did not read back.
        named = [
            f"nothing to save: {', '.join(only_defaults)} used only default values"
            if only_defaults
            else "nothing to save"
        ]
        raise PresetError("; ".join([*named, *notes]))
    try:
        preset = Preset(
            stand=stand, plate=plate, wells=wells, saved_utc=saved_utc, params=params
        )
    except ValidationError as exc:
        first = " ".join(
            str(exc.errors()[0]["msg"]).removeprefix("Value error, ").split()
        )
        raise PresetError(f"nothing saved: {first}") from None
    return PresetFromRun(preset, tuple(notes))


def _pairs_of(
    plugin: str, model: type[PluginParams] | None, recorded: Mapping[str, Any]
) -> tuple[tuple[str, ...], str | None]:
    """The ``key=value`` pairs a record's parameters leave in a preset, or why there are none.

    ``recorded`` is data taken as it comes (FM-125): the plugin may have changed
    since the run, so the ``Params`` validates it again. A pair is saved when its
    field is required or differs from its default, in the model's declaration
    order, except the keys a preset holds elsewhere (``NOT_IN_PARAMS``).

    The pairs are then **read back** the way ``smc run --preset`` reads them, with
    what the run takes from elsewhere (the plate, the wells, the operator's
    positions) given as the record has it, and the result must equal the record
    key by key. A text that reads back as another value (the string ``null`` in a
    ``str | None`` field) would make a later run use a value the operator never
    chose, and nothing would say so.
    """
    if model is None:
        return (), (
            f"{plugin}: no plugin of that name is loaded; its parameters are not saved"
        )
    try:
        validated = model.model_validate(recorded)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'parameters'}: "
            f"{error['msg']}"
            for error in exc.errors()
        )
        return (), (
            f"{plugin}: its recorded parameters are not valid for this smc "
            f"({problems}); not saved"
        )
    values = validated.model_dump(mode="json")
    fields = model.model_fields
    held = NOT_IN_PARAMS.get(plugin, {})
    items = [
        f"{key}={format_param(field, values[key])}"
        for key, field in fields.items()
        if key not in held
        and (
            field.is_required()
            or values[key]
            != to_jsonable_python(field.get_default(call_default_factory=True))
        )
    ]
    if not items:
        return (), None
    elsewhere = [
        f"{key}={format_param(fields[key], values[key])}"
        for key in held
        if key in fields
    ]
    try:
        back = parse_params(model, [*items, *elsewhere]).model_dump(mode="json")
    except ParamsError as exc:
        return (), (
            f"{plugin}: its pairs do not read back ({'; '.join(exc.problems)}); "
            "not saved"
        )
    changed = [key for key in fields if back[key] != values[key]]
    if changed:
        return (), (
            f"{plugin}: {', '.join(changed)} would read back as another value; "
            "not saved"
        )
    for item in items:
        if not _is_text(item):
            return (), (
                f"{plugin}: {item.partition('=')[0]} holds a character that is not "
                "text; not saved"
            )
    return tuple(items), None


def _is_text(text: str) -> bool:
    """Whether ``text`` encodes as UTF-8: a lone surrogate, left by a label read in cp1252, does not (FM-05)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
