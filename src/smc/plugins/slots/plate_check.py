"""The plate-check slot: when a registered plate was last confirmed where it was registered (m3 §27.4).

At Ti2-E session 5 (#550, F15) a plate registered by ``plate-walls`` was lifted
and put back, and nothing noticed: a stage position stays true, what goes stale
is its reading against the wells. ``plate-check``, and the scan before its first
field, measure two registered wells again under the objective in place and
compare them with the registered centres moved into that objective's frame
(§18). The record is the answer to "when was this registration last
confirmed": ``checked_utc``. A refused check has no result; its reason carries
every number.

Every position here is a stage position (``XYStage.position_um()``), never an
image side (FM-189).
"""

from __future__ import annotations

from pydantic import AwareDatetime, Field

from smc.plugins.api import PluginResult
from smc.plugins.slots.base import SlotValue
from smc.plugins.slots.plate import WellFit

__all__ = ["CheckedWell", "PlateCheck", "PlateCheckResult"]


class CheckedWell(SlotValue):
    """One registered well measured again: where it was registered, where it was expected, where it is.

    ``registered_xy_um`` is the registration's measured centre under the plate's
    own objective; ``expected_xy_um`` is it plus ``PlateCheck.shift_um``, the
    same well in the frame of the objective in place. ``fit`` is the walls' own
    measure, and ``difference_xy_um`` its centre minus ``expected_xy_um``, whose
    length is ``distance_um``: how far the well moved since it was registered.
    """

    well: str
    registered_xy_um: tuple[float, float]
    expected_xy_um: tuple[float, float]
    fit: WellFit
    difference_xy_um: tuple[float, float]
    distance_um: float = Field(ge=0.0)


class PlateCheck(SlotValue):
    """A check that passed: the wells measured, the bound they were held to, and when.

    ``plate_objective`` is the objective the plate was registered with and
    ``objective`` the one the check was made under (``None`` without a turret);
    ``shift_um`` is §18's translation between their frames, ``(0, 0)`` under the
    plate's own objective. ``worst_um`` is the largest ``distance_um``, at most
    ``max_shift_um``. ``turn_deg`` and ``slide_xy_um`` are the rigid motion that
    carries the expected centres onto the measured ones: said, never judged.
    ``frames`` counts every frame the walls took; ``folder`` is run-relative
    (``plate-check/<token>``), ``None`` when frames are not saved. ``elapsed_s``
    runs from the first move to the last well, and ``checked_utc`` is the clock
    when the last well was measured.
    """

    checked_utc: AwareDatetime
    plate: str
    plate_objective: str | None
    objective: str | None
    shift_um: tuple[float, float]
    wells: tuple[CheckedWell, ...]
    worst_um: float = Field(ge=0.0)
    max_shift_um: float = Field(gt=0.0)
    turn_deg: float
    slide_xy_um: tuple[float, float]
    frames: int = Field(ge=0)
    folder: str | None
    elapsed_s: float = Field(ge=0.0)


class PlateCheckResult(PluginResult):
    """What ``plate-check`` fills its slot with: the check that passed."""

    check: PlateCheck
