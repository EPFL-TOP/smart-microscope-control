"""Interchangeable microscope tools: the plugin API, its registry and the built-in plugins (M3).

A plugin is a class (``Plugin``) registered under the ``smc.plugins`` entry-point
group; ``discover``/``load`` find it, ``run_plugin`` runs it on a ``Microscope``
and records the outcome in the run document (``smc.run``). Plugins depend on the
capability Protocols of ``smc.hardware`` and on nothing else of it (ADR-0003,
ADR-0004).
"""

from smc.plugins.api import (
    NAME_PATTERN,
    Plugin,
    PluginParams,
    PluginResult,
    Progress,
    RunContext,
    StopRequested,
)
from smc.plugins.errors import (
    ParamsError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    RequirementsError,
    ResumeError,
    UpstreamError,
)
from smc.plugins.params import describe_params, parse_params
from smc.plugins.registry import (
    GROUP,
    LoadFailure,
    PluginEntry,
    Registry,
    discover,
    load,
)
from smc.plugins.runner import check_ready, run_plugin

__all__ = [
    "GROUP",
    "NAME_PATTERN",
    "LoadFailure",
    "ParamsError",
    "Plugin",
    "PluginEntry",
    "PluginError",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginParams",
    "PluginResult",
    "Progress",
    "Registry",
    "RequirementsError",
    "ResumeError",
    "RunContext",
    "StopRequested",
    "UpstreamError",
    "check_ready",
    "describe_params",
    "discover",
    "load",
    "parse_params",
    "run_plugin",
]
