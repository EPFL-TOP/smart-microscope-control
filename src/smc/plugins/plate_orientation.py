"""``plate-orientation`` says which way the registered plate lies, and renames its wells (m3 §24 D7).

A plate whose wells form a uniform grid lands on itself when it is turned 180°:
the positions stay and only the names swap. When the operator learns after a fit
that the plate lies the other way (a well's printed label, the cross on the page),
nothing needs measuring again: this tool reads the latest *other* provider of
``plate`` and returns it declared ``orientation``, through
``PlateResult.with_orientation``. It is the command line's way to say it, and the
page's switch runs it (ADR-0006: nothing exists only in the UI).

It reads the slot it provides. The run document leaves a reader's own record out
of every read (m3 §24 D8), so it never reads its own previous output and is not
stale by construction; the tools after it read its result like any ``plate``, and
are stale once it runs, which is right: their well names changed. A fit run again
later is the newer provider, and the turn is set aside with no extra rule.

The orientation has no default: running the tool is a deliberate statement. It is
absolute, not a toggle: the fit's own orientation gives the fit's numbers back.
It touches no device and writes no file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from smc.plugins.api import PluginParams, RunContext
from smc.plugins.slots import PlateResult
from smc.plugins.slots.plate import PlateOrientation

if TYPE_CHECKING:
    from smc.hardware.microscope import Microscope

__all__ = ["PlateOrientationParams", "PlateOrientationPlugin"]


class PlateOrientationParams(PluginParams):
    """Which way the plate lies; required, so that a run says it on purpose."""

    orientation: PlateOrientation = Field(
        description=(
            "Which way the plate lies: not-turned or turned-180. The latest "
            "plate fit is renamed to it; nothing is measured again."
        )
    )


class PlateOrientationPlugin:
    """Renames the registered plate's wells for the orientation the operator declares."""

    name = "plate-orientation"
    summary = "Say which way the registered plate lies; renames its wells."
    requires: tuple[type, ...] = ()
    reads = ("plate",)
    reads_optional: tuple[str, ...] = ()
    provides = "plate"
    Params = PlateOrientationParams
    Result = PlateResult

    def run(
        self,
        microscope: Microscope,
        params: PlateOrientationParams,
        ctx: RunContext,
    ) -> PlateResult:
        """The latest other plate fit, declared ``params.orientation``.

        Raises:
            UpstreamError: The fit's result does not validate as a plate.
        """
        return ctx.upstream("plate", PlateResult).with_orientation(params.orientation)
