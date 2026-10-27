"""The base of every value inside a slot's result (m3 §3).

It lives in its own module and not in ``slots/__init__.py``: the slot modules
import it, and ``__init__`` imports the slot modules, so a base defined in
``__init__`` fails with a circular import (measured, #199).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["SlotValue"]


class SlotValue(BaseModel):
    """A value inside a slot's result: frozen, strict about keys, finite.

    The config is ``PluginResult``'s, so a value nested in a result is held to
    the same rules as the result: a key this smc does not know is refused and
    never dropped (FM-107), and ``nan`` and ``inf`` are refused in every float
    (FM-32), because unknown is ``None`` and never ``nan``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
