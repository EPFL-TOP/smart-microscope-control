"""Plugin parameters from text, and their description (m3 §1.6).

``smc run --param key=value`` and a future form in a UI post the same strings, so
the conversion is here and not in the CLI. Pydantic's lax mode already turns the
text of a scalar into its type (``"3"`` into ``3``, ``"no"`` into ``False``, a
path, an enum member), measured in §1.1. It does not turn text into a list or a
tuple, so a container field takes JSON. The model's own bounds and the base
class's ``allow_inf_nan=False`` do the validating (FM-32), and every problem is
collected before one ``ParamsError`` lists them all: an operator who mistyped
two keys should not have to run the command twice to find the second.
"""

from __future__ import annotations

import json
import types
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import ValidationError
from pydantic.fields import FieldInfo

from smc.plugins.api import P, PluginParams
from smc.plugins.errors import ParamsError

__all__ = ["describe_params", "parse_params"]

_SCALARS: tuple[type, ...] = (str, int, float, bool, Path)

#: The bounds ``describe_params`` shows, as ``annotated_types`` names them.
_CONSTRAINTS = ("gt", "ge", "lt", "le", "multiple_of", "min_length", "max_length")

_JSON_EXAMPLE = '["A1", "B2"]'

#: What ``_decode`` returns for text that is not JSON where JSON is wanted.
_NOT_JSON: Any = object()


def parse_params(model: type[P], pairs: Sequence[str]) -> P:
    """Build ``model`` from ``key=value`` items; every problem is reported at once.

    Each item splits at its first ``=``, so a value may contain one and may be
    empty. A scalar field takes the text as it is (``null`` is ``None`` when the
    field admits it); any other field takes JSON. Keys are the field names
    exactly: there are no aliases.

    The error names the model's class, which is all this function knows of the
    plugin: the CLI raises it again with the plugin's name.

    Raises:
        ParamsError: An item is not ``key=value``, a key is unknown or given
            twice, a value is not JSON where JSON is wanted, or the model
            refuses the result (a bound, a type, ``nan``, a missing field).
    """
    fields = model.model_fields
    problems: list[str] = []
    raw: dict[str, Any] = {}
    unusable: set[str] = set()  # keys already reported: pydantic must not repeat them
    for item in pairs:
        key, separator, value = item.partition("=")
        if not separator or not key:
            problems.append(f"expected key=value, got '{item}'")
        elif key not in fields:
            problems.append(f"unknown key '{key}' (keys: {', '.join(fields)})")
        elif key in raw or key in unusable:
            problems.append(f"'{key}' given twice")
        else:
            decoded = _decode(fields[key], value)
            if decoded is _NOT_JSON:
                problems.append(f"'{key}' takes JSON, for example {_JSON_EXAMPLE}")
                unusable.add(key)
            else:
                raw[key] = decoded
    # Validated even when a key was wrong, so that a bad value is reported beside
    # it; the result is only returned when nothing was.
    try:
        parsed = model.model_validate(raw)
    except ValidationError as exc:
        problems.extend(_validation_problems(exc, unusable))
    else:
        if not problems:
            return parsed
    raise ParamsError(model.__name__, tuple(problems))


def _validation_problems(error: ValidationError, unusable: set[str]) -> list[str]:
    found: list[str] = []
    for item in error.errors():
        location = item["loc"]
        if location and str(location[0]) in unusable:
            continue
        where = ".".join(str(part) for part in location) or "parameters"
        found.append(f"{where}: {item['msg']}")
    return found


def _decode(field: FieldInfo, text: str) -> Any:
    """The value to hand to pydantic for ``text``: the text itself, ``None``, or decoded JSON."""
    annotation, admits_none = _without_none(field.annotation)
    if _is_scalar(annotation):
        return None if admits_none and text == "null" else text
    try:
        return json.loads(text)
    except ValueError:
        return _NOT_JSON


def _without_none(annotation: Any) -> tuple[Any, bool]:
    """The annotation of a field with ``None`` taken out of its union, and whether it was there."""
    if get_origin(annotation) in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        admits_none = len(members) != len(get_args(annotation))
        if len(members) == 1:
            return members[0], admits_none
        return annotation, admits_none
    return annotation, False


def _is_scalar(annotation: Any) -> bool:
    """``str``, ``int``, ``float``, ``bool``, ``Path``, an ``Enum`` or a ``Literal``."""
    if get_origin(annotation) is Literal:
        return True
    if not isinstance(annotation, type):
        return False
    return annotation in _SCALARS or issubclass(annotation, Enum)


def describe_params(model: type[PluginParams]) -> list[tuple[str, str, str, str]]:
    """One row per field, in declaration order: name, type, default, constraints and description.

    What ``smc run NAME --show-params`` prints, derived from the model, so the
    keys it lists are the keys ``parse_params`` accepts. The default is
    ``(required)`` when there is none; the constraints are the ``annotated_types``
    bounds of the field (``ge=1 le=100``), followed by its description.
    """
    rows: list[tuple[str, str, str, str]] = []
    for name, field in model.model_fields.items():
        default = (
            "(required)"
            if field.is_required()
            else repr(field.get_default(call_default_factory=True))
        )
        notes = [_constraints(field), field.description or ""]
        rows.append(
            (
                name,
                _type_name(field.annotation),
                default,
                "  ".join(n for n in notes if n),
            )
        )
    return rows


def _type_name(annotation: Any) -> str:
    """A type as it reads in the source: every class by its ``__name__``, ``typing.`` left out.

    Built piece by piece, not from ``str(annotation)``: that spells ``Path`` as
    ``pathlib._local.Path`` on Python 3.13 and ``pathlib.Path`` before it.
    """
    if annotation is type(None):
        return "None"
    if annotation is Ellipsis:
        return "..."
    origin = get_origin(annotation)
    if origin is None:
        return annotation.__name__ if isinstance(annotation, type) else repr(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(arg) for arg in args)
    if origin is Literal:
        return f"Literal[{', '.join(repr(arg) for arg in args)}]"
    name = getattr(origin, "__name__", str(origin))
    return f"{name}[{', '.join(_type_name(arg) for arg in args)}]"


def _constraints(field: FieldInfo) -> str:
    found: list[str] = []
    for meta in field.metadata:
        for constraint in _CONSTRAINTS:
            value = getattr(meta, constraint, None)
            if value is not None:
                found.append(f"{constraint}={value!r}")
    return " ".join(found)
