"""Finding the selectors and the orderers: two entry-point groups, checked, never trusted (algorithms design §5.4, §5.8).

An algorithm is a class in a package that is installed next to smc, or in a
collaborator's own checkout, and it is found by its entry point's name. That
code was not reviewed here and runs inside ``route``, so the registry follows the
plugin registry's rule (m3 §1.5): one that fails to import, names the wrong
class or breaks a rule is a **finding** and the others load (FM-113), never the
end of the listing.

Three decisions the measurements made (design §5.8):

* **A name is looked up without importing anything.** ``RouteParams()`` is built
  on every parameter check, and ``importlib.metadata.entry_points`` costs 2 to
  5 ms a call, so ``selector_names()`` reads only the entry points' names and
  ``selector(name)`` loads the entries of that name and no other. A cache would
  hide a test's replacement of ``entry_points``, so there is none.
* **Entry points are metadata written when smc is installed** (FM-112). A venv
  that was not reinstalled after a new algorithm was registered does not know it:
  when no entry is registered at all, the message says to run ``pip install -e
  .``.
* **A finding is text, never the exception** (FM-98): a kept exception keeps its
  traceback, and with it the frames and the locals of the plugin that raised it.

``KeyboardInterrupt`` and ``SystemExit`` raised by an import pass through: an
operator's Ctrl-C is not an algorithm's finding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any, cast

from smc.plugins.api import NAME_PATTERN
from smc.selection.api import (
    Orderer,
    OrdererParams,
    SelectionError,
    Selector,
    SelectorParams,
)

__all__ = [
    "ORDERER_GROUP",
    "SELECTOR_GROUP",
    "orderer",
    "orderer_names",
    "orderers",
    "selector",
    "selector_names",
    "selectors",
]

SELECTOR_GROUP = "smc.selectors"
ORDERER_GROUP = "smc.orderers"

_log = logging.getLogger("smc.selection")


@dataclass(frozen=True, slots=True)
class _Kind:
    """What differs between the two groups: the noun in the messages, the group, the base of ``Params`` and the method."""

    noun: str
    group: str
    params_base: type
    method: str


_SELECTOR = _Kind("selector", SELECTOR_GROUP, SelectorParams, "select")
_ORDERER = _Kind("orderer", ORDERER_GROUP, OrdererParams, "order")


def selector_names() -> tuple[str, ...]:
    """The registered selectors' names, sorted; nothing is imported."""
    return tuple(sorted(_by_name(_SELECTOR)))


def orderer_names() -> tuple[str, ...]:
    """The registered orderers' names, sorted; nothing is imported."""
    return tuple(sorted(_by_name(_ORDERER)))


def selectors() -> dict[str, Selector[Any]]:
    """Every selector that loads and passes the checks, by name, sorted; each that does not is a logged warning."""
    return cast("dict[str, Selector[Any]]", _all(_SELECTOR))


def orderers() -> dict[str, Orderer[Any]]:
    """Every orderer that loads and passes the checks, by name, sorted; each that does not is a logged warning."""
    return cast("dict[str, Orderer[Any]]", _all(_ORDERER))


def selector(name: str) -> Selector[Any]:
    """The selector registered as ``name``: only its entries are imported.

    Raises:
        SelectionError: No such name (the known ones are listed, sorted; with
            none registered at all the text says to reinstall), two entries with
            different targets, or the entry fails to load or breaks a rule (the
            reason is in the text).
    """
    return cast("Selector[Any]", _one(_SELECTOR, name))


def orderer(name: str) -> Orderer[Any]:
    """The orderer registered as ``name``; see ``selector``.

    Raises:
        SelectionError: As for ``selector``.
    """
    return cast("Orderer[Any]", _one(_ORDERER, name))


def _by_name(kind: _Kind) -> dict[str, list[EntryPoint]]:
    """The group's entry points by name, as installed; this module's ``entry_points``, so that a test can put its own in place."""
    grouped: dict[str, list[EntryPoint]] = {}
    for ep in entry_points(group=kind.group):
        grouped.setdefault(ep.name, []).append(ep)
    return grouped


def _all(kind: _Kind) -> dict[str, object]:
    found: dict[str, object] = {}
    by_name = _by_name(kind)
    for name in sorted(by_name):
        instance, reason = _load_name(kind, by_name[name])
        if reason is None:
            found[name] = instance
        else:
            _log.warning("%s %r is left out: %s", kind.noun, name, reason)
    return found


def _one(kind: _Kind, name: str) -> object:
    by_name = _by_name(kind)
    if name not in by_name:
        raise SelectionError(_unknown(kind, name, sorted(by_name)))
    instance, reason = _load_name(kind, by_name[name])
    if reason is not None:
        raise SelectionError(f"{kind.noun} {name!r} {reason}")
    return instance


def _unknown(kind: _Kind, name: str, known: list[str]) -> str:
    if not known:
        return (
            f"no {kind.noun} is registered: run pip install -e . in the smc "
            f"folder (entry points are read when smc is installed)"
        )
    return f"no {kind.noun} named {name!r}; known: {', '.join(known)}"


def _load_name(kind: _Kind, entries: list[EntryPoint]) -> tuple[object, str | None]:
    """The one algorithm an entry name stands for, or why it is not one: ``(instance, None)`` or ``(None, reason)``.

    An editable install can list one distribution twice, so identical values
    are one entry; two different targets for one name are both refused, so
    nothing shadows an algorithm silently.
    """
    distinct: dict[str, EntryPoint] = {}
    for ep in entries:
        distinct.setdefault(ep.value, ep)
    if len(distinct) > 1:
        listed = ", ".join(
            f"{value} ({_distribution(ep) or 'no distribution'})"
            for value, ep in sorted(distinct.items())
        )
        return None, f"is registered twice, with different targets: {listed}"
    (ep,) = distinct.values()
    return _inspect(kind, ep)


def _distribution(ep: EntryPoint) -> str | None:
    """The name of the distribution that registered ``ep``, when known (a hand-built entry point has no ``dist``, and on Python 3.10 not the attribute)."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None
    return str(dist.metadata["Name"])


def _inspect(kind: _Kind, ep: EntryPoint) -> tuple[object, str | None]:
    """Import ``ep``, create it and check the rules; the first that fails is the reason.

    Every step is inside one guard that turns an ``Exception`` into text: a
    package on a microscope PC is code nobody here reviewed, and a property that
    raises when read, a constructor that needs a file, an import that fails are
    findings like any other (FM-113). The phase it failed in is named.
    """
    phase = "imported"
    try:
        target = ep.load()
        if not isinstance(target, type):
            return None, f"names a {type(target).__name__}, not a class"
        phase = "created with no argument"
        instance = target()
        phase = "inspected"
        problem = _problem(kind, instance, ep.name)
    except Exception as exc:
        # Text, never the exception: a kept one keeps its traceback (FM-98).
        reason = f"could not be {phase}: {_text_of(exc)}"
    else:
        if problem is None:
            return instance, None
        reason = problem
    return None, reason


def _problem(kind: _Kind, instance: Any, entry_name: str) -> str | None:
    """The first rule the instance breaks: the members, the name, the summary, the parameters, the method; ``None`` when it breaks none.

    A member that raises when it is read is not caught here: ``_inspect`` turns
    it into a finding (``inspected: RuntimeError: ...``).
    """
    missing = [
        member
        for member in ("name", "summary", "Params", kind.method)
        if not hasattr(instance, member)
    ]
    if missing:
        return f"lacks {', '.join(missing)}"
    name = instance.name
    if not isinstance(name, str):
        return f"has a name that is not a string: {name!r}"
    if name != entry_name:
        return (
            f"is named {name!r}, but its entry point is named {entry_name!r}; "
            f"they must be equal"
        )
    if not re.fullmatch(NAME_PATTERN, name):
        return (
            f"is named {name!r}, which is not kebab-case (lowercase letters and "
            f"digits, single hyphens between them, starting with a letter)"
        )
    summary = instance.summary
    if not isinstance(summary, str):
        return f"has a summary that is not a string: {summary!r}"
    if not summary.strip():
        return "has an empty summary: say in one line what it does"
    if "\n" in summary.strip() or "\r" in summary:
        return "has a summary of more than one line"
    params = instance.Params
    if not (isinstance(params, type) and issubclass(params, kind.params_base)):
        shown = params.__name__ if isinstance(params, type) else repr(params)
        return f"has Params {shown}, which is not a subclass of {kind.params_base.__name__}"
    if not callable(getattr(instance, kind.method)):
        return f"has a {kind.method} that is not callable"
    return None


def _text_of(error: BaseException) -> str:
    """``Type: message`` on one line."""
    return f"{type(error).__name__}: {' '.join(str(error).split())}"
