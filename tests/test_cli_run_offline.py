"""``smc run NAME --offline``: the plugins that never drive the stand, on a copied run folder (algorithms §4, #566).

The runner pins the width (FM-37) and clears ``SMC_PROFILE`` and ``SMC_PROFILES``,
so an operator's environment never reaches a test (FM-40). Every command here runs
under ``no_stand``: no microscope may be opened, from a profile or a core, and no
Micro-Manager adapter is found, so a command that opened a stand and swallowed the
``CoreError`` would still fail. Test plugins are classes of this module, put in
the registry through ``entry_points``; the five installed offline plugins are
looked at through the registry as installed.
"""

from __future__ import annotations

from typing import Any

from smc.plugins import discover
from smc.plugins.api import Plugin


def _installed() -> dict[str, Plugin[Any, Any]]:
    return {name: entry.plugin for name, entry in discover().plugins.items()}


# --- which plugins are offline -----------------------------------------------


def test_the_five_plugins_that_never_drive_the_stand_are_the_offline_ones() -> None:
    plugins = _installed()

    offline = sorted(
        name for name, plugin in plugins.items() if getattr(plugin, "offline", False)
    )

    assert offline == ["cells", "density", "nis-export", "report", "route"]
    # The attribute is a real True, as the command reads it (`is True`).
    assert all(plugins[name].offline is True for name in offline)  # type: ignore[attr-defined]


def test_an_offline_plugin_requires_no_capability() -> None:
    # A plugin that sets `offline` runs on a facade that has no capability: one
    # that required any would fail on every offline run (api.Plugin's docstring).
    for name, plugin in _installed().items():
        if getattr(plugin, "offline", False):
            assert plugin.requires == (), name
