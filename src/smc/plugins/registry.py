"""Finding the plugins: entry points, checked, never trusted (m3 §1.5).

Plugins are discovered through the ``smc.plugins`` entry-point group, so that a
third-party package can add one without touching this repository (ADR-0004). A
package on a microscope PC is code nobody here reviewed, and it is imported when
``smc run --list`` runs: one that fails to import, names the wrong class or lacks
a member must not end the listing of the others. Every such entry is a
``LoadFailure`` with the reason, a finding in the sense of M1 §2, and the rest
load.

``load(name)`` imports only the entries with that name, so ``smc run
stage-report`` never imports a package that is broken for another reason.
``KeyboardInterrupt`` and ``SystemExit`` raised by an import pass through: an
operator's Ctrl-C is not a plugin's finding.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any, cast

from smc.hardware.microscope import CAPABILITY_ROLES
from smc.plugins.api import NAME_PATTERN, Plugin, PluginParams, PluginResult
from smc.plugins.errors import PluginLoadError, PluginNotFoundError

__all__ = [
    "GROUP",
    "LoadFailure",
    "PluginEntry",
    "Registry",
    "discover",
    "load",
]

GROUP = "smc.plugins"

#: The members of the ``Plugin`` protocol, in the order it declares them: the
#: ones a plugin that fails ``isinstance`` is named as lacking.
_MEMBERS = (
    "name",
    "summary",
    "Params",
    "Result",
    "requires",
    "provides",
    "reads",
    "reads_optional",
    "run",
)

_SLOT_HINT = "kebab-case, as a plugin name"

#: A member set to ``None`` is present: only a member that is not there at all is lacking.
_ABSENT = object()


@dataclass(frozen=True, slots=True)
class PluginEntry:
    """A plugin that passed every check, with where it came from."""

    plugin: Plugin[Any, Any]
    target: str
    distribution: str | None
    version: str | None


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """An entry point that is not a plugin, and why: a finding, not an exception."""

    name: str
    target: str
    distribution: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Registry:
    """What ``discover`` found: the plugins by name, and the entries that failed."""

    plugins: Mapping[str, PluginEntry]
    failures: tuple[LoadFailure, ...]


def discover(entry_points: Iterable[EntryPoint] | None = None) -> Registry:
    """Import and check every entry point of the group.

    Never raises for a plugin: each entry that fails a check becomes a
    ``LoadFailure`` and the others load. ``entry_points=None`` reads the
    installed distributions; a test passes its own.

    Returns:
        The plugins ordered by name, and the failures by name, then target.
    """
    by_name = _by_name(_listed(entry_points))
    plugins: dict[str, PluginEntry] = {}
    failures: list[LoadFailure] = []
    for name in sorted(by_name):
        outcome = _load_name(name, by_name[name])
        if isinstance(outcome, PluginEntry):
            plugins[name] = outcome
        else:
            failures.extend(outcome)
    failures.sort(key=lambda failure: (failure.name, failure.target))
    return Registry(plugins=plugins, failures=tuple(failures))


def load(name: str, entry_points: Iterable[EntryPoint] | None = None) -> PluginEntry:
    """Import and check the entries named ``name``, and no other.

    Raises:
        PluginNotFoundError: No entry has this name; the error lists the names
            of the group, which were read without importing anything.
        PluginLoadError: The entry failed a check; it holds the
            ``LoadFailure``.
    """
    by_name = _by_name(_listed(entry_points))
    if name not in by_name:
        raise PluginNotFoundError(name, tuple(sorted(by_name)))
    outcome = _load_name(name, by_name[name])
    if isinstance(outcome, PluginEntry):
        return outcome
    raise PluginLoadError(outcome[0])


def _listed(given: Iterable[EntryPoint] | None) -> list[EntryPoint]:
    """The entry points to check: the ones given, or the group as installed.

    The installed group is read through this module's ``entry_points``, so a CLI
    test can put its own plugins in place of the installed ones.
    """
    if given is None:
        return list(entry_points(group=GROUP))
    return list(given)


def _by_name(entry_points: list[EntryPoint]) -> dict[str, list[EntryPoint]]:
    grouped: dict[str, list[EntryPoint]] = {}
    for ep in entry_points:
        grouped.setdefault(ep.name, []).append(ep)
    return grouped


def _distribution(ep: EntryPoint) -> tuple[str | None, str | None]:
    """The name and version of the distribution that registered ``ep``, when known.

    ``getattr``: an ``EntryPoint`` built by hand has no ``dist``, and on Python
    3.10 not even the attribute.
    """
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None, None
    return dist.metadata["Name"], dist.version


def _load_name(name: str, entries: list[EntryPoint]) -> PluginEntry | list[LoadFailure]:
    """The one plugin an entry name stands for, or the findings about it.

    An editable install can list one distribution twice, so identical
    ``(name, value)`` pairs are one entry; two different targets for one name
    are both refused, so nothing shadows a plugin silently.
    """
    distinct: dict[str, EntryPoint] = {}
    for ep in entries:
        distinct.setdefault(ep.value, ep)
    if len(distinct) > 1:
        listed = ", ".join(
            f"{value} ({_distribution(ep)[0] or 'no distribution'})"
            for value, ep in sorted(distinct.items())
        )
        return [
            LoadFailure(
                name, value, _distribution(ep)[0], f"registered twice: {listed}"
            )
            for value, ep in sorted(distinct.items())
        ]
    (ep,) = distinct.values()
    return _check(ep)


def _check(ep: EntryPoint) -> PluginEntry | list[LoadFailure]:
    """Import ``ep`` and run the checks of §1.5 in order; the first that fails is the finding."""
    distribution, version = _distribution(ep)

    def failed(reason: str) -> list[LoadFailure]:
        return [LoadFailure(ep.name, ep.value, distribution, reason)]

    try:
        target = ep.load()
    except Exception as exc:
        # Text, never the error: a kept finding would keep its traceback (FM-98).
        return failed(f"could not be imported: {_text_of(exc)}")
    if not isinstance(target, type):
        return failed(f"names a {type(target).__name__}, not a class")
    try:
        instance = target()
    except Exception as exc:
        return failed(f"could not be created: {_text_of(exc)}")
    try:
        problem = _problem(instance, ep.name)
    except Exception as exc:
        # A member that raises when it is read (a property that needs a file) is
        # a finding like any other, not the end of `smc run --list` (FM-79). On
        # Python before 3.12 even `isinstance` reads it, through `hasattr`.
        return failed(f"could not be inspected: {_text_of(exc)}")
    if problem is not None:
        return failed(problem)
    return PluginEntry(
        plugin=cast("Plugin[Any, Any]", instance),
        target=ep.value,
        distribution=distribution,
        version=version,
    )


def _problem(instance: object, entry_name: str) -> str | None:
    """Checks 4 to 6 in order: the members, the names, the models; the first problem."""
    if not isinstance(instance, Plugin):
        missing = [
            member
            for member in _MEMBERS
            if inspect.getattr_static(instance, member, _ABSENT) is _ABSENT
            and not hasattr(instance, member)
        ]
        return f"lacks {', '.join(missing)}" if missing else "is not a Plugin"
    return _naming_problem(instance, entry_name) or _model_problem(instance)


def _text_of(error: BaseException) -> str:
    """``Type: message`` on one line."""
    return f"{type(error).__name__}: {' '.join(str(error).split())}"


def _naming_problem(plugin: Plugin[Any, Any], entry_name: str) -> str | None:
    """Check 5: the name, the summary, the slot it provides and the slots it reads."""
    # `summary` is read by `smc run --list` for every plugin: a value that is not
    # text would end the listing of the others.
    for field in ("name", "summary", "provides"):
        value = getattr(plugin, field)
        if not isinstance(value, str):
            return f"{field} must be a string, got {value!r}"
    if plugin.name != entry_name:
        return (
            f"is named {plugin.name!r}, but its entry point is named "
            f"{entry_name!r}; they must be equal"
        )
    if not re.fullmatch(NAME_PATTERN, plugin.name):
        return (
            f"is named {plugin.name!r}, which is not kebab-case (lowercase letters "
            f"and digits, single hyphens between them, starting with a letter)"
        )
    if not re.fullmatch(NAME_PATTERN, plugin.provides):
        return f"provides {plugin.provides!r}, which is not a slot name ({_SLOT_HINT})"
    for field in ("reads", "reads_optional"):
        slots = getattr(plugin, field)
        if not isinstance(slots, tuple) or not all(isinstance(s, str) for s in slots):
            return f"{field} must be a tuple of slot names, got {slots!r}"
        for slot in slots:
            if not re.fullmatch(NAME_PATTERN, slot):
                return f"{field} {slot!r}, which is not a slot name ({_SLOT_HINT})"
        if plugin.provides in slots:
            return f"{field} holds {plugin.provides!r}, the slot it provides itself"
    shared = [slot for slot in plugin.reads if slot in plugin.reads_optional]
    if shared:
        return f"lists {', '.join(shared)} in both reads and reads_optional"
    return None


def _model_problem(plugin: Plugin[Any, Any]) -> str | None:
    """Check 6: the two models and the capabilities it requires."""
    for field, base in (("Params", PluginParams), ("Result", PluginResult)):
        model = getattr(plugin, field)
        if not (isinstance(model, type) and issubclass(model, base)):
            shown = model.__name__ if isinstance(model, type) else repr(model)
            return f"{field} {shown} is not a subclass of {base.__name__}"
    # Declared ``object``: the protocol types it as a tuple, which is what is checked.
    requires: object = plugin.requires
    if not isinstance(requires, tuple):
        return f"requires must be a tuple of capability classes, got {_names(requires)}"
    for capability in requires:
        if not (isinstance(capability, type) and capability in CAPABILITY_ROLES):
            shown = (
                capability.__name__
                if isinstance(capability, type)
                else repr(capability)
            )
            return f"requires {shown}, which is not a capability"
    return None


def _names(value: object) -> str:
    """``repr`` of ``value``, with the classes of a list or tuple shown by their name."""
    if isinstance(value, list | tuple):
        inner = ", ".join(
            item.__name__ if isinstance(item, type) else repr(item) for item in value
        )
        return f"[{inner}]" if isinstance(value, list) else f"({inner})"
    return repr(value)
