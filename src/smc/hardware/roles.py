"""Roles: what a device is *for* (design: docs/design/m1-hardware-layer.md §4).

A tool asks for "the XY stage", never for ``"XYStage"`` or ``"TIXYDrive"``;
this module decides which loaded device fills each role. The order of
authority is ADR-0003's: the profile's explicit mapping (a human said so),
then MMCore's own role slots (the configuration said so), then heuristics.

The heuristics are **type-first**, because the device type is the one thing
every adapter reports reliably, and names only choose between devices of
the same type. Both Nikon stands show why names alone fail and types alone
are not enough:

* the Ti2 reports **four** ``XYStage`` devices: the stage and three TIRF
  illuminator positioners. Driving a TIRF positioner reads exactly like a
  dead stage: it reports 0,0 and never moves;
* the Ti-E reports **three** ``Stage`` devices: the Z drive, the PFS offset
  (which is not in microns) and the TIRF drive.

So each role has a rule: the types that can fill it, name fragments that a
candidate must contain, fragments that exclude it for good, and hints that
rank the survivors. Name matching is on the normalised *label* (``[a-z0-9]``
only), because the label is what the ``.cfg`` and the core use; the
adapter's device name only breaks ties. An exclusion also overrides the
configuration: a ``.cfg`` written before the TIRF exclusion existed may
still name ``TIRF1`` as the XY stage, and it is corrected with a warning
instead of silently driving the wrong axis.

Every candidate is reported, best first, because a role with several
candidates is where a wrong pick hides.

Nothing here imports ``pymmcore_plus`` at module level: ``resolve()`` is pure
and testable on plain ``DeviceInfo`` lists, and only the two core helpers
touch a core.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

__all__ = [
    "DeviceInfo",
    "Role",
    "RoleMap",
    "Source",
    "core_roles",
    "devices_from_core",
    "resolve",
]

log = logging.getLogger("smc.hardware.roles")


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


Source = Literal["profile", "core", "heuristic"]


@dataclass(frozen=True, slots=True)
class Rule:
    """How the heuristics recognise the devices that can fill one role.

    Fragments are normalised (``[a-z0-9]`` only) and matched as substrings of
    the normalised label. A device is a candidate when its type is one of
    ``types``, its label contains one of ``require_any`` (if any are given)
    and none of ``exclude``. Candidates rank by the first ``hints`` entry
    their label contains; the device name breaks ties, then the label.
    """

    types: tuple[str, ...]
    require_any: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    hints: tuple[str, ...] = ()


# The §4 table. Generalised from nikon-control's scope/stand.py; every
# exclusion is a device that fits the type and must never fill the role.
_RULES: dict[Role, Rule] = {
    Role.camera: Rule(types=("Camera",)),
    # The Ti2's three TIRF positioners are XYStage devices too.
    Role.xy_stage: Rule(
        types=("XYStage",),
        exclude=("tirf",),
        hints=("xystage", "xydrive", "stage"),
    ),
    # The PFS offset and the TIRF drive are Stage devices, and neither
    # moves the objective.
    Role.focus: Rule(
        types=("Stage",),
        exclude=("pfs", "offset", "tirf"),
        hints=("zdrive", "focus", "z"),
    ),
    Role.autofocus: Rule(types=("AutoFocus",)),
    Role.autofocus_offset: Rule(
        types=("Stage",),
        require_any=("pfs", "offset"),
        hints=("pfsoffset", "offset"),
    ),
    # Filter, condenser and reflector turrets are State devices named
    # "...Turret" as well.
    Role.objective_turret: Rule(
        types=("State",),
        require_any=("nose", "objective", "turret"),
        exclude=("filter", "condenser", "reflector"),
        hints=("nosepiece", "objective", "turret"),
    ),
    Role.shutter: Rule(
        types=("Shutter",),
        hints=("epishutter", "epi", "diashutter", "dia", "tl"),
    ),
    # A device named "...Shutter" only gates the light; the light source is
    # the device that sets how bright it is. Revisited in M2 with real
    # Nikon and Zeiss device lists.
    Role.light_source: Rule(
        types=("Shutter", "State", "Generic"),
        require_any=("lamp", "led", "light", "laser", "colibri"),
        exclude=("shutter",),
        hints=("lamp", "led", "laser"),
    ),
    Role.light_path: Rule(
        types=("State",),
        require_any=("lightpath", "sideport", "port", "eyepiece"),
        hints=("lightpath", "sideport"),
    ),
    Role.filter_turret: Rule(
        types=("State",),
        require_any=("filter", "reflector"),
        exclude=("objective", "nose"),
        hints=("filter", "reflector"),
    ),
}

# The .cfg property that sets each MMCore slot (``Property,Core,XYStage,XY``),
# so a warning can say which line to fix.
_CORE_SLOTS: dict[Role, str] = {
    Role.camera: "Camera",
    Role.xy_stage: "XYStage",
    Role.focus: "Focus",
    Role.autofocus: "AutoFocus",
    Role.shutter: "Shutter",
}


@dataclass
class RoleMap:
    """Which device fills each role, where that came from, and what else could have.

    A role with no device is absent from ``assigned``; it is never mapped to
    ``""``, which would read as present everywhere downstream.
    """

    assigned: dict[Role, str]
    sources: dict[Role, Source]
    candidates: dict[Role, list[str]]  # best first; includes the assigned one
    warnings: list[str]

    def get(self, role: Role) -> str | None:
        """The label assigned to ``role``, or ``None`` when nothing fills it."""
        return self.assigned.get(role)

    def missing(self) -> list[Role]:
        """The roles nothing fills, in ``Role`` order."""
        return [role for role in Role if role not in self.assigned]

    def ambiguous(self) -> dict[Role, list[str]]:
        """Roles with more than one candidate: where a wrong pick would hide."""
        return {
            role: list(labels)
            for role, labels in self.candidates.items()
            if len(labels) > 1
        }

    def describe(self) -> list[str]:
        """One aligned line per role, in ``Role`` order; the CLI prints it verbatim.

        ``xy stage          XY  (also: Stage2)  [core]``: the role, the
        device (``-`` when nothing fills it), the runner-ups and the source.
        """
        rows: list[list[str]] = []
        for role in Role:
            label = self.assigned.get(role)
            others = [c for c in self.candidates.get(role, []) if c != label]
            rows.append(
                [
                    role.value.replace("_", " "),
                    label if label is not None else "-",
                    f"(also: {', '.join(others)})" if others else "",
                    f"[{self.sources[role]}]" if role in self.sources else "",
                ]
            )
        # Drop a column nobody uses, so its padding does not show as a gap.
        used = [i for i in range(4) if i < 2 or any(row[i] for row in rows)]
        widths = {i: max(len(row[i]) for row in rows) for i in used}
        return [
            "  ".join(row[i].ljust(widths[i]) for i in used).rstrip() for row in rows
        ]


def _norm(text: str) -> str:
    """Lower-case and keep ``[a-z0-9]`` only, so ``"TI ZDrive"`` matches ``zdrive``."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _first_hit(text: str, fragments: Iterable[str]) -> str | None:
    """The first fragment ``text`` contains, or ``None``."""
    return next((f for f in fragments if f in text), None)


def _rank(text: str, hints: Sequence[str]) -> int:
    """Position of the first hint ``text`` contains; lower is better."""
    return next((i for i, hint in enumerate(hints) if hint in text), len(hints))


def _is_candidate(device: DeviceInfo, rule: Rule, exclude: Sequence[str]) -> bool:
    if device.type not in rule.types:
        return False
    label = _norm(device.label)
    if rule.require_any and _first_hit(label, rule.require_any) is None:
        return False
    return _first_hit(label, exclude) is None


def _ranked(
    devices: Sequence[DeviceInfo], rule: Rule, exclude: Sequence[str]
) -> list[str]:
    found = [d for d in devices if _is_candidate(d, rule, exclude)]
    found.sort(
        key=lambda d: (
            _rank(_norm(d.label), rule.hints),
            _rank(_norm(d.name), rule.hints),
            d.label.casefold(),
            d.label,
        )
    )
    return [d.label for d in found]


def _exclusions(
    role: Role, rule: Rule, extra: Sequence[str]
) -> tuple[list[str], list[str]]:
    """The built-in fragments plus the profile's, normalised; and any warnings."""
    exclude = list(rule.exclude)
    warnings: list[str] = []
    # A bare string is a Sequence[str] too: iterated, it would exclude every
    # label containing any one of its letters.
    for fragment in (extra,) if isinstance(extra, str) else extra:
        if _norm(fragment):
            exclude.append(_norm(fragment))
        else:
            # An empty fragment is contained in every label: it would exclude
            # every device, which no profile author means.
            warnings.append(
                f"{role.value}: ignoring the exclusion {fragment!r} in "
                f"[roles.exclude], which has no letter or digit to match."
            )
    return exclude, warnings


def _choose(
    role: Role,
    ranked: Sequence[str],
    exclude: Sequence[str],
    loaded: Mapping[str, DeviceInfo],
    override: str | None,
    slot_label: str | None,
) -> tuple[tuple[str, Source] | None, list[str]]:
    """The first source that yields a device, and a warning per source passed over."""
    rule = _RULES[role]
    choice: tuple[str, Source] | None = None
    notes: list[str] = []
    passed_over: list[str] = []

    if override is not None:
        target = loaded.get(override)
        if target is None:
            passed_over.append(
                f"[roles.assign] names {override!r}, which is not a loaded device "
                f"(labels are case-sensitive; candidates: {', '.join(ranked) or 'none'})"
            )
        else:
            choice = (override, "profile")
            if target.type not in rule.types:
                notes.append(
                    f"{role.value}: [roles.assign] names {override!r}, a "
                    f"{target.type} device, but {role.value} is filled by "
                    f"{' or '.join(rule.types)} devices; using it as assigned."
                )

    if choice is None and slot_label:
        slot = _CORE_SLOTS.get(role, role.value)
        hit = _first_hit(_norm(slot_label), exclude)
        if slot_label not in loaded:
            passed_over.append(
                f"the core's {slot} slot names {slot_label!r}, which is not "
                f"among the loaded devices"
            )
        elif hit is not None:
            passed_over.append(
                f"the configuration names {slot_label!r}, which never fills "
                f"{role.value} (its label contains {hit!r}); fix "
                f"Property,Core,{slot} in the .cfg or set [roles.assign] {role.value}"
            )
        else:
            choice = (slot_label, "core")

    if choice is None and ranked:
        choice = (ranked[0], "heuristic")

    outcome = (
        f"using {choice[0]!r} ({choice[1]})" if choice else "leaving it unassigned"
    )
    notes.extend(f"{role.value}: {reason}; {outcome}." for reason in passed_over)
    return choice, notes


def resolve(
    devices: Iterable[DeviceInfo],
    *,
    core_roles: Mapping[Role, str] | None = None,
    overrides: Mapping[Role, str] | None = None,
    exclusions: Mapping[Role, Sequence[str]] | None = None,
) -> RoleMap:
    """Decide which device fills each role, and report every other candidate.

    For each role, the first source that yields a device wins:

    1. ``overrides[role]`` (the profile's ``[roles.assign]``): honoured even
       for a device the rules exclude, because a human said so. A label
       that is not loaded is a warning, and resolution falls through.
    2. ``core_roles[role]`` (MMCore's own slots): refused with a warning
       when the device is excluded for the role, e.g. a ``.cfg`` that still
       names a TIRF positioner as the XY stage.
    3. The best-ranked heuristic candidate.

    Args:
        devices: The loaded devices (see :func:`devices_from_core`). Labels
            are unique in a core; a repeated label keeps its first entry.
        core_roles: The core's slots (see :func:`core_roles`).
        overrides: The profile's ``[roles.assign]``.
        exclusions: The profile's ``[roles.exclude]``: extra fragments per
            role, normalised like labels and added to the built-in ones.

    Returns:
        The assignment, its sources, the candidates per role (every role is
        a key; the assigned label comes first) and the warnings, in
        ``Role`` order.
    """
    loaded: dict[str, DeviceInfo] = {}
    for device in devices:
        if device.label and device.label not in loaded:
            loaded[device.label] = device
    listed = list(loaded.values())
    slots = core_roles or {}
    wanted = overrides or {}
    extra = exclusions or {}

    assigned: dict[Role, str] = {}
    sources: dict[Role, Source] = {}
    candidates: dict[Role, list[str]] = {}
    warnings: list[str] = []
    for role in Role:
        rule = _RULES[role]
        exclude, notes = _exclusions(role, rule, extra.get(role, ()))
        ranked = _ranked(listed, rule, exclude)
        choice, more = _choose(
            role, ranked, exclude, loaded, wanted.get(role), slots.get(role)
        )
        warnings += notes + more
        if choice is not None:
            assigned[role], sources[role] = choice
            ranked = [choice[0], *(c for c in ranked if c != choice[0])]
        candidates[role] = ranked

    return RoleMap(
        assigned=assigned, sources=sources, candidates=candidates, warnings=warnings
    )


class _InventoryCore(Protocol):
    """The part of a core the two helpers read.

    Structural, so that ``CMMCorePlus`` and the ``FakeCore`` of the tests
    both qualify without one inheriting from the other.
    """

    def getLoadedDevices(self) -> Sequence[str]: ...  # noqa: N802 - MMCore API
    def getDeviceType(self, label: str, /) -> int: ...  # noqa: N802
    def getDeviceLibrary(self, label: str, /) -> str: ...  # noqa: N802
    def getDeviceName(self, label: str, /) -> str: ...  # noqa: N802
    def getDeviceDescription(self, label: str, /) -> str: ...  # noqa: N802
    def getCameraDevice(self) -> str: ...  # noqa: N802
    def getXYStageDevice(self) -> str: ...  # noqa: N802
    def getFocusDevice(self) -> str: ...  # noqa: N802
    def getAutoFocusDevice(self) -> str: ...  # noqa: N802
    def getShutterDevice(self) -> str: ...  # noqa: N802


def devices_from_core(core: _InventoryCore) -> list[DeviceInfo]:
    """Every loaded device except ``Core``, as the resolver sees it.

    A device whose type this pymmcore-plus does not know is listed with type
    ``"Unknown"`` and fills no role: one exotic device must not end the
    inventory of the whole stand.
    """
    from pymmcore_plus import DeviceType

    found: list[DeviceInfo] = []
    for label in core.getLoadedDevices():
        if label == "Core":
            continue
        try:
            kind = DeviceType(core.getDeviceType(label)).name
        except ValueError as exc:
            log.warning("%s: unknown device type (%s); it fills no role", label, exc)
            kind = "Unknown"
        found.append(
            DeviceInfo(
                label=label,
                type=kind.removesuffix("Device"),
                library=core.getDeviceLibrary(label),
                name=core.getDeviceName(label),
                description=core.getDeviceDescription(label),
            )
        )
    return found


def core_roles(core: _InventoryCore) -> dict[Role, str]:
    """The roles MMCore itself has slots for, where the configuration filled them."""
    slots = {
        Role.camera: core.getCameraDevice(),
        Role.xy_stage: core.getXYStageDevice(),
        Role.focus: core.getFocusDevice(),
        Role.autofocus: core.getAutoFocusDevice(),
        Role.shutter: core.getShutterDevice(),
    }
    return {role: label for role, label in slots.items() if label}
