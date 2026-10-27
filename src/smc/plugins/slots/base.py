"""The base of every value inside a slot's result (m3 §3).

It lives in its own module and not in ``slots/__init__.py``: the slot modules
import it, and ``__init__`` imports the slot modules, so a base defined in
``__init__`` fails with a circular import (measured, #199).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["SlotValue", "refuse_non_finite"]


class SlotValue(BaseModel):
    """A value inside a slot's result: frozen, strict about keys, finite.

    The config is ``PluginResult``'s, so a value nested in a result is held to
    the same rules as the result: a key this smc does not know is refused and
    never dropped (FM-107), and ``nan`` and ``inf`` are refused in every float
    (FM-32), because unknown is ``None`` and never ``nan``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def refuse_non_finite(value: Any, what: str) -> None:
    """Raise ``ValueError`` if a float inside ``value`` is ``nan`` or infinite.

    For a field typed with another library's model (``useq``'s plans), whose
    floats the slot model's ``allow_inf_nan=False`` does not reach. ``value``
    is a dumped model: dicts, lists and tuples of scalars. The message names
    one offender by its path, so the operator can find it in the file.
    """
    stack: list[tuple[str, Any]] = [(what, value)]
    while stack:
        path, item = stack.pop()
        if isinstance(item, dict):
            stack.extend((f"{path}.{key}", child) for key, child in item.items())
        elif isinstance(item, list | tuple):
            stack.extend((f"{path}[{i}]", child) for i, child in enumerate(item))
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{path} must be finite, got {item!r}")
