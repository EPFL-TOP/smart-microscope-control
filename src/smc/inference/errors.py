"""The one error of the inference package: a model that cannot be used here."""

from __future__ import annotations

__all__ = ["ModelError"]


class ModelError(Exception):
    """A model that cannot be used here: the message says why and what to do."""
