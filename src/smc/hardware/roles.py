"""Roles: what a device is *for* (design: docs/design/m1-hardware-layer.md §4).

Seeded by the design PR with the two types every wave-1 issue shares.
``RoleMap``, ``resolve()`` and the core inventory helpers arrive with #6.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """The roles the control layer asks for. Values double as profile keys."""

    camera = "camera"
    xy_stage = "xy_stage"
    focus = "focus"
    autofocus = "autofocus"
    autofocus_offset = "autofocus_offset"
    objective_turret = "objective_turret"
    shutter = "shutter"
    light_source = "light_source"
    light_path = "light_path"
    filter_turret = "filter_turret"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """A loaded device as the resolver sees it: label first, type second.

    ``type`` is Micro-Manager's ``DeviceType`` name without the ``Device``
    suffix (``"XYStage"``, ``"Stage"``, ``"Camera"``, ``"State"``,
    ``"Shutter"``, ``"AutoFocus"``, ``"Hub"``, ``"Generic"``...).
    """

    label: str
    type: str
    library: str = ""
    name: str = ""
    description: str = ""
