"""Errors of the plugin layer (m3 §1.9).

A plugin that cannot be found, loaded, configured or started is a problem of
the command line or the stand's profile, not a device fault, so none of these
is a ``HardwareError`` or a ``RuntimeError``: a plugin's ``except HardwareError``
never takes one, and ``smc run`` maps them to its own exit codes. Every message
names the plugin and ends with what to do, because it is read at the stand by
someone who did not write the plugin.

Each class with its own ``__init__`` defines ``__reduce__``, as
``smc.hardware.errors`` does, so that it survives a worker process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smc.plugins.registry import LoadFailure

__all__ = [
    "ParamsError",
    "PluginError",
    "PluginLoadError",
    "PluginNotFoundError",
    "RequirementsError",
    "ResumeError",
    "UpstreamError",
]


class PluginError(Exception):
    """A plugin could not be found, loaded, configured or started."""


class PluginNotFoundError(PluginError):
    """No plugin of the group has this name; ``available`` lists the ones there are."""

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = available
        if available:
            message = (
                f"no plugin named {name!r} (available: {', '.join(available)}); "
                f"smc run --list shows them"
            )
        else:
            # Plugins are registered by metadata that pip writes at install time:
            # a PC that was only `git pull`ed has none until it is reinstalled.
            message = (
                f"no plugin named {name!r}, and none is registered. If smc was "
                f"updated without reinstalling, run pip install -e . in the smc "
                f"folder (plugins are registered when it is installed)"
            )
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.name, self.available))


class PluginLoadError(PluginError):
    """The plugin's entry point failed one of the registry's checks (§1.5)."""

    def __init__(self, failure: LoadFailure) -> None:
        self.failure = failure
        origin = f", from {failure.distribution}" if failure.distribution else ""
        super().__init__(
            f"plugin {failure.name} ({failure.target}{origin}) {failure.reason}; "
            f"smc run --list shows the plugins that do load"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.failure,))


class ParamsError(PluginError):
    """``--param`` did not make valid parameters; every problem is listed at once."""

    def __init__(self, plugin: str, problems: tuple[str, ...]) -> None:
        self.plugin = plugin
        self.problems = problems
        super().__init__(
            f"parameters of {plugin}: {'; '.join(problems)}. "
            f"smc run {plugin} --show-params lists the keys"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.plugin, self.problems))


class RequirementsError(PluginError):
    """The stand lacks capabilities the plugin requires; nothing was written.

    ``details`` holds each missing capability's own ``CapabilityMissingError``
    message, which points at ``[roles.assign]`` in the profile (FM-36: the
    brackets are part of the advice).
    """

    def __init__(
        self, plugin: str, missing: tuple[str, ...], details: tuple[str, ...]
    ) -> None:
        self.plugin = plugin
        self.missing = missing
        self.details = details
        super().__init__(
            f"{plugin} needs {' and '.join(missing)}, which this stand does not "
            f"have: {' '.join(details)}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.plugin, self.missing, self.details))


class UpstreamError(PluginError):
    """A slot the plugin reads cannot be used; ``reason`` ends with the fix."""

    def __init__(self, plugin: str, slot: str, reason: str) -> None:
        self.plugin = plugin
        self.slot = slot
        self.reason = reason
        super().__init__(f"{plugin} reads {slot}, but {reason}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.plugin, self.slot, self.reason))


class ResumeError(PluginError):
    """The stage cannot be resumed (or its checkpoint cannot be read); ``reason`` says why."""

    def __init__(self, plugin: str, reason: str) -> None:
        self.plugin = plugin
        self.reason = reason
        super().__init__(f"cannot resume {plugin}: {reason}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.plugin, self.reason))
