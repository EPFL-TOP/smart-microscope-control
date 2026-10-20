"""A Micro-Manager configuration from the devices that answered (design m2 §10).

Pure: nothing here imports ``pymmcore_plus`` or reads hardware. A
:class:`~smc.discovery.models.BuildScan` (what the build child found) and the
file being replaced go in; a plan and the text of the new file come out, so
every rule is a unit test on values built by hand.

The rule that shapes the module is design §10.5: **every line of the previous
file that the build does not write itself comes back**, carried, or set aside as
a comment that the next build reads again. ``nikon-control``'s rebuild deleted
every line outside eight directives, typed state labels included, and carried
presets that named devices the new file no longer loaded (FM-107). Here a
preset, a pixel-size resolution, a device's labels and a ``Delay`` or
``FocusDirection`` line are each carried or set aside **whole**, never in part
(FM-82, FM-99), and a set-aside line is written as ``# not carried: <line>``,
which :func:`parse_previous` reads as that line again.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from smc.discovery.models import BuildScan, BuiltDevice, ProbedDevice, ProbeResult
from smc.hardware.roles import DeviceInfo, Role, RoleMap, resolve

__all__ = [
    "FIELD_COUNTS",
    "NOT_CARRIED",
    "Block",
    "ConfigPlan",
    "PreviousConfig",
    "cut_short",
    "parse_previous",
    "plan_config",
    "read_previous",
    "render_config",
]

#: What a set-aside line starts with. The rest of the line is read, by the next
#: build, as a line of the file.
NOT_CARRIED = "# not carried: "

#: How many non-empty fields each directive this MMCore reads has, the directive
#: included (design m2 §10.1). MMCore drops empty fields, then refuses a line
#: whose count is wrong, and the whole load fails (``Label,D,0,a,b``): a line
#: of one of these directives with another count is set aside, never carried.
FIELD_COUNTS: Mapping[str, frozenset[int]] = {
    "ConfigGroup": frozenset({2, 5, 6}),
    "ConfigPixelSize": frozenset({5}),
    "PixelSize_um": frozenset({3}),
    "PixelSizeAffine": frozenset({8}),
    "PixelSizeAngle_dxdz": frozenset({3}),
    "PixelSizeAngle_dydz": frozenset({3}),
    "PixelSizeOptimalZ_Um": frozenset({3}),
    "Label": frozenset({4}),
    "Delay": frozenset({3}),
    "FocusDirection": frozenset({3}),
}

#: The pixel-size directives that carry no device: a resolution's number.
_PIXEL_VALUES = frozenset(
    {
        "PixelSize_um",
        "PixelSizeAffine",
        "PixelSizeAngle_dxdz",
        "PixelSizeAngle_dydz",
        "PixelSizeOptimalZ_Um",
    }
)

#: ``Property,Core,<slot>,<label>``: the slot of MMCore, and the role it holds.
_SLOT_ROLES: Mapping[str, Role] = {
    "Camera": Role.camera,
    "XYStage": Role.xy_stage,
    "Focus": Role.focus,
    "AutoFocus": Role.autofocus,
    "Shutter": Role.shutter,
}

#: The lines of the Roles section, in the order the file writes them, and the
#: device type each slot accepts: MMCore refuses a Core slot set to a device of
#: another type, and the whole load fails.
_SLOT_ORDER: tuple[tuple[Role, str, str], ...] = (
    (Role.camera, "Camera", "Camera"),
    (Role.shutter, "Shutter", "Shutter"),
    (Role.focus, "Focus", "Stage"),
    (Role.xy_stage, "XYStage", "XYStage"),
    (Role.autofocus, "AutoFocus", "AutoFocus"),
)

#: ``Core`` settings whose value is a device (a preset that sets one names it).
_CORE_DEVICE_SETTINGS = frozenset(
    {
        "Camera",
        "Shutter",
        "Focus",
        "XYStage",
        "AutoFocus",
        "Galvo",
        "SLM",
        "ImageProcessor",
    }
)

#: ``Property,Core,<these>``: written by the build itself, so not read back.
_WRITTEN_CORE_PROPERTIES = frozenset({"Initialize", "AutoShutter"})

BlockKind = Literal["preset", "pixel", "labels", "device", "property", "other"]


@dataclass(frozen=True, slots=True)
class Block:
    """The lines of the previous file that are carried or set aside together."""

    kind: BlockKind
    #: ``Channel/DAPI``, ``Res10x``, ``Nosepiece``, ``Delay ZDrive``, …
    key: str
    #: Verbatim, in file order.
    lines: tuple[str, ...]
    #: The labels it names, ``Core`` excluded.
    devices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreviousConfig:
    """The file being replaced, read as blocks."""

    path: Path | None
    #: In order of first appearance.
    blocks: tuple[Block, ...]
    #: ``Property,Core,<Camera|XYStage|Focus|AutoFocus|Shutter>,<label>``.
    core_slots: Mapping[Role, str]


@dataclass
class ConfigPlan:
    """What the file will say, and why anything is missing from it."""

    adapter: str
    camera_adapter: str | None
    #: What the file loads, in the order written.
    devices: list[BuiltDevice]
    roles: RoleMap
    carried: list[Block]
    #: The block, and why.
    not_carried: list[tuple[Block, str]]
    #: ``<adapter>/<device>``, and why.
    left_out: list[tuple[str, str]]
    #: Not empty: nothing may be written.
    refusals: list[str]
    notes: list[str]

    @property
    def writable(self) -> bool:
        """Whether a file may be written: the plan has no refusal."""
        return not self.refusals


# --- the previous file ---------------------------------------------------------------


def _fields(line: str) -> list[str]:
    """The non-empty, stripped fields of a line, as MMCore counts them."""
    return [field for field in (f.strip() for f in line.split(",")) if field]


def _unique(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def _named(device: str, prop: str, value: str) -> list[str]:
    """The devices a setting names: ``Core`` names the device it is set to."""
    if device != "Core":
        return [device]
    return [value] if prop in _CORE_DEVICE_SETTINGS and value else []


@dataclass
class _Draft:
    kind: BlockKind
    key: str
    lines: list[str]
    devices: list[str]


def parse_previous(text: str, path: Path | None = None) -> PreviousConfig:
    r"""Read a Micro-Manager ``.cfg`` as the blocks a rebuild carries or sets aside.

    Drops a byte-order mark at the start (FM-80), blank lines and comments, except
    a line that starts with :data:`NOT_CARRIED`: its rest is read as a line of the
    file, so a block set aside by an earlier build comes back once it applies
    again. What the build writes itself (``Device``, ``Parent``, the Core's
    ``Initialize`` and ``AutoShutter``, and the five slots, which go to
    ``core_slots``) is dropped. Lines are split on ``\n`` only, a ``\r`` before
    it dropped, as MMCore reads them: a ``\x85`` or U+2028 inside a label is
    part of the line, and comes back the way it was.
    """
    if text.startswith("\N{ZERO WIDTH NO-BREAK SPACE}"):
        text = text[1:]
    drafts: list[_Draft] = []
    merged: dict[tuple[str, str], _Draft] = {}
    core_slots: dict[Role, str] = {}

    def add(
        kind: BlockKind,
        key: str,
        line: str,
        devices: Sequence[str] = (),
        *,
        merge: bool = False,
    ) -> None:
        draft = merged.get((kind, key)) if merge else None
        if draft is None:
            draft = _Draft(kind, key, [], [])
            drafts.append(draft)
            if merge:
                merged[(kind, key)] = draft
        draft.lines.append(line)
        draft.devices.extend(devices)

    for raw in text.split("\n"):
        line = raw[:-1] if raw.endswith("\r") else raw
        if line.startswith(NOT_CARRIED):
            line = line[len(NOT_CARRIED) :]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = _fields(line)
        if not fields:
            continue  # only commas: it says nothing (a blank row saved from Excel)
        directive, count = fields[0], len(fields)

        if directive in ("Device", "Parent") and count >= 2:
            # Dropped by the plan when this build declares that device itself;
            # kept as a comment when it does not (another adapter's, FM-107).
            add("device", f"{directive} {fields[1]}", line, [fields[1]])
            continue
        if directive == "Property":
            if count >= 2 and fields[1] == "Core":
                slot = fields[2] if count > 2 else ""
                value = fields[3] if count > 3 else ""
                if slot in _SLOT_ROLES:
                    if value:
                        core_slots[_SLOT_ROLES[slot]] = value
                elif slot == "AutoShutter":
                    # The build writes AutoShutter,1 with a shutter. Another value
                    # was somebody's choice, and is not the build's to overwrite
                    # without a word.
                    if value and value != "1":
                        add("property", "Property Core AutoShutter", line)
                elif slot not in _WRITTEN_CORE_PROPERTIES and value:
                    # A Core setting the build does not write, and a line that
                    # sets nothing is nothing to keep.
                    add("property", f"Property Core {slot}", line)
            else:
                device = fields[1] if count >= 2 else ""
                add(
                    "property",
                    " ".join(fields[:3]),
                    line,
                    [device] if device else [],
                )
            continue
        if count not in FIELD_COUNTS.get(directive, frozenset()):
            add("other", " ".join(line.split()), line)
        elif directive == "ConfigGroup":
            group = fields[1]
            if count == 2:
                add("preset", group, line, merge=True)
            else:
                value = fields[5] if count == 6 else ""
                add(
                    "preset",
                    f"{group}/{fields[2]}",
                    line,
                    _named(fields[3], fields[4], value),
                    merge=True,
                )
        elif directive == "ConfigPixelSize":
            add(
                "pixel",
                fields[1],
                line,
                _named(fields[2], fields[3], fields[4]),
                merge=True,
            )
        elif directive in _PIXEL_VALUES:
            add("pixel", fields[1], line, merge=True)
        elif directive == "Label":
            add("labels", fields[1], line, [fields[1]], merge=True)
        else:  # Delay, FocusDirection: one line each
            add("device", f"{directive} {fields[1]}", line, [fields[1]])

    blocks = tuple(
        Block(d.kind, d.key, tuple(d.lines), _unique(d.devices)) for d in drafts
    )
    return PreviousConfig(path=path, blocks=blocks, core_slots=core_slots)


def read_previous(path: Path) -> PreviousConfig | None:
    r"""The file at ``path``, read byte for byte; ``None`` when there is none.

    Read as UTF-8 with ``errors="surrogateescape"``, and without translating
    line ends, so that the bytes of a line it carries are written back unchanged
    (a cp1252 ``µ`` stays one byte) and a lone ``\r`` does not become a line
    break MMCore does not see.

    Raises:
        OSError: The file exists and cannot be read.
    """
    try:
        with path.open(encoding="utf-8", errors="surrogateescape", newline="") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    return parse_previous(text, path)


# --- the plan ------------------------------------------------------------------------


def _unwritable_reason(text: str) -> str | None:
    """Why ``text`` cannot be a field of a ``.cfg`` line, or ``None``.

    A comma splits the field, and a line break (or any control character, or a
    separator that ``str.splitlines`` takes for one: ``stands_in_config`` reads
    files that way) ends the line, so what follows would be read as a directive.
    """
    if not text.strip():
        return "it is empty"
    if "," in text:
        return "it holds a comma"
    if any(
        unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in text
    ):  # \n, \r, NUL, \x85, U+2028
        return "it holds a line break or a control character"
    return None


def _why_not(row: ProbedDevice) -> str:
    """What the end of the file says about a device that did not initialise."""
    if row.status == "failed":
        error = row.error or "failed"
        return f"{error} - {row.hint}" if row.hint else error
    reason = row.reason or "not probed"
    if reason.startswith("needs a port"):
        # The walk's own reason names --port, an option this command lacks.
        return "needs a serial port, which smc config build does not set"
    return reason


def _hub_failures(probe: ProbeResult) -> list[str]:
    return [
        f"hub {row.name} of {probe.adapter} did not initialise"
        for row in probe.devices
        if row.type == "Hub" and row.status == "failed"
    ]


def cut_short(scan: BuildScan) -> bool:
    """Whether the rows are the ones a child that died had finished.

    A scan marked whole whose walks are not was put together from checkpoints:
    its rows stop where the child did, and nothing says that what is missing was
    not there.
    """
    walks = (scan.stand, scan.camera)
    return not scan.complete or any(w is not None and not w.complete for w in walks)


def _refusals(scan: BuildScan) -> list[str]:
    """Why nothing may be written, one line each (design m2 §10.4, rule 1)."""
    if cut_short(scan):
        return ["the build child stopped before the end"]
    out: list[str] = []
    stand = scan.stand
    if stand.error is not None:
        out.append(f"{stand.adapter}: {stand.error}")
    # A failed hub refuses even when another hub then initialised: the Ti2-E's
    # simulator answers when the stand is off, and a file written from it would
    # load a stand that is not there. A hub left out with --skip is not failed.
    hub_failures = _hub_failures(stand)
    out.extend(hub_failures)
    if not (stand.error is not None or hub_failures) and not any(
        row.status == "ok" and row.type != "Hub" for row in stand.devices
    ):
        what = (
            "nothing below the hub initialised"
            if any(row.type == "Hub" for row in stand.devices)
            else "no device initialised"
        )
        out.append(f"{stand.adapter}: {what}")
    camera = scan.camera
    if camera is not None:
        camera_failures = _hub_failures(camera)
        if camera.error is not None:
            out.append(f"{camera.adapter}: {camera.error}")
        out.extend(camera_failures)
        if not (camera.error is not None or camera_failures) and not any(
            row.status == "ok" and row.type == "Camera" for row in camera.devices
        ):
            out.append(f"{camera.adapter}: no camera initialised")
    return out


def _select_devices(
    scan: BuildScan,
) -> tuple[list[BuiltDevice], list[tuple[str, str]]]:
    """The devices the file loads, and the ones it leaves out (rule 2)."""
    built = {(d.library, d.label): d for d in scan.devices}
    camera = scan.camera
    camera_hubs: set[str] = set()
    if camera is not None:
        for row in camera.devices:
            device = built.get((camera.adapter, row.name))
            if (
                row.status == "ok"
                and row.type == "Camera"
                and device is not None
                and device.parent
            ):
                camera_hubs.add(device.parent)

    # (key, device or None, why not), in the order of the rows.
    entries: list[tuple[str, BuiltDevice | None, str | None]] = []
    walks = [scan.stand] if camera is None else [scan.stand, camera]
    for walk in walks:
        for row in walk.devices:
            key = f"{walk.adapter}/{row.name}"
            device = built.get((walk.adapter, row.name))
            if row.status != "ok":
                entries.append((key, None, _why_not(row)))
            elif device is None:
                entries.append((key, None, "the build child did not read it"))
            elif (
                walk is camera and row.type != "Camera" and row.name not in camera_hubs
            ):
                entries.append(
                    (key, None, "not a camera: --camera-adapter adds cameras only")
                )
            elif (
                _unwritable_reason(row.name) is not None or row.name != row.name.strip()
            ):
                entries.append((key, None, "its name cannot be written in a .cfg line"))
            else:
                entries.append((key, device, None))

    # A device whose hub is left out cannot be loaded: its Parent line would name
    # a device the file does not load. Repeated, since a hub may itself be left out.
    kept = {d.label for _, d, _ in entries if d is not None}
    dropped: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for key, device, _ in entries:
            if device is None or key in dropped:
                continue
            if device.parent is not None and device.parent not in kept:
                dropped[key] = f"its hub {device.parent} is left out"
                kept.discard(device.label)
                changed = True

    written = [d for key, d, _ in entries if d is not None and key not in dropped]
    left_out: list[tuple[str, str]] = []
    for key, device, why in entries:
        if device is None:
            left_out.append((key, why or ""))
        elif key in dropped:
            left_out.append((key, dropped[key]))
    return written, left_out


def _core_slots(
    roles: RoleMap, devices: Sequence[BuiltDevice]
) -> tuple[list[tuple[str, str]], list[str]]:
    """``Property,Core,<slot>,<label>`` for each assigned role, and the notes.

    A role the profile assigned to a device of another type is warned about by
    ``resolve`` and honoured there; here it would make MMCore refuse the whole
    file (a Core slot takes one type of device), so the slot is left unset and
    the note says so.
    """
    types = {d.label: d.type for d in devices}
    slots: list[tuple[str, str]] = []
    notes: list[str] = []
    for role, slot, wanted in _SLOT_ORDER:
        label = roles.get(role)
        if label is None:
            continue
        if types.get(label) != wanted:
            notes.append(
                f"{role.value}: {label} is a {types.get(label)} device, not a "
                f"{wanted} one: no Property,Core,{slot} line was written"
            )
            continue
        slots.append((slot, label))
    return slots, notes


def _label_settings(block: Block) -> list[tuple[str, str]]:
    """The ``(device, label)`` pairs a preset or a pixel-size block selects."""
    found: list[tuple[str, str]] = []
    for line in block.lines:
        fields = _fields(line)
        if block.kind == "preset" and len(fields) == 6 and fields[4] == "Label":
            found.append((fields[3], fields[5]))
        elif (
            block.kind == "pixel"
            and fields[0] == "ConfigPixelSize"
            and len(fields) == 5
            and fields[3] == "Label"
        ):
            found.append((fields[2], fields[4]))
    return found


def _join(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


_DEVICE_PROPERTY = (
    "a device setting: the build writes no Property line for a device; put it in a "
    "preset (System/Startup)"
)
_CORE_PROPERTY = (
    "a Core setting the build does not write (it writes Initialize, AutoShutter and "
    "the five slots); put it in a preset (System/Startup)"
)
_AUTO_SHUTTER = (
    "the build writes Property,Core,AutoShutter,1 whenever a shutter is assigned; "
    "to keep another value, put it in a preset (System/Startup)"
)
_UNWALKED = (
    "a device this build did not initialise (another adapter's, or one that is "
    "gone): it is not loaded; put the line back by hand if the stand needs it"
)
_UNREAD = "not a line this Micro-Manager reads"


def _is_declaration(block: Block) -> bool:
    """Whether the block is a ``Device`` or ``Parent`` line."""
    return block.kind == "device" and block.key.split(" ", 1)[0] in ("Device", "Parent")


def _split_previous(
    previous: PreviousConfig,
    written: Mapping[str, BuiltDevice],
    own_labels: Mapping[str, list[str]],
    walked: Collection[str],
) -> tuple[list[Block], list[tuple[Block, str]]]:
    """Rule 5: each block of the previous file is carried, or set aside whole.

    ``walked`` names every device of both walks, whatever became of it: the
    build declares those itself (or lists them as left out), so their old
    ``Device`` and ``Parent`` lines are not read back.
    """
    # The previous file's Label lines, per device, replace the device's own for
    # that device only (rule 4): all or nothing, the positions were named by hand.
    kept_labels: dict[str, set[str]] = {}
    carried: list[Block] = []
    not_carried: list[tuple[Block, str]] = []

    def aside(block: Block, why: str) -> None:
        not_carried.append((block, why))

    for block in previous.blocks:
        if block.kind != "labels":
            continue
        (device,) = block.devices
        if device not in written:
            aside(block, f"names {device}, which this configuration does not load")
        elif written[device].type != "State":
            aside(
                block,
                f"{device} is a {written[device].type} device here, which has no labels",
            )
        else:
            carried.append(block)
            kept_labels[device] = {_fields(line)[3] for line in block.lines}

    def labels_of(device: str) -> set[str]:
        return kept_labels.get(device, set(own_labels.get(device, ())))

    for block in previous.blocks:
        if block.kind == "labels":
            continue
        if _is_declaration(block):
            # What this build declares itself comes out of it again (a device
            # that did not initialise is listed at the end); a device it did not
            # walk has no line in the new file, so the old one is kept as a
            # comment: the next build reads it, and a human can put it back.
            if block.devices[0] not in walked:
                aside(block, _UNWALKED)
        elif block.kind == "property":
            if block.key == "Property Core AutoShutter":
                aside(block, _AUTO_SHUTTER)
            else:
                aside(block, _DEVICE_PROPERTY if block.devices else _CORE_PROPERTY)
        elif block.kind == "other":
            aside(block, _UNREAD)
        else:
            absent = [d for d in block.devices if d not in written]
            if absent:
                aside(
                    block,
                    f"names {_join(absent)}, which this configuration does not load",
                )
                continue
            wrong = next(
                (
                    (device, label)
                    for device, label in _label_settings(block)
                    if label not in labels_of(device)
                ),
                None,
            )
            if wrong is not None:
                aside(
                    block,
                    f"sets {wrong[0]} to the label {wrong[1]}, which it does not "
                    "have here",
                )
            else:
                carried.append(block)
    # Carried blocks come back in the previous file's order, the labels included.
    order = {id(block): i for i, block in enumerate(previous.blocks)}
    carried.sort(key=lambda block: order[id(block)])
    not_carried.sort(key=lambda item: order[id(item[0])])
    return carried, not_carried


def plan_config(
    scan: BuildScan,
    *,
    previous: PreviousConfig | None = None,
    overrides: Mapping[Role, str] | None = None,
    exclusions: Mapping[Role, Sequence[str]] | None = None,
    skip: Sequence[str] = (),
) -> ConfigPlan:
    """Decide what the file loads, what it carries over, and what it must not do.

    The rules, in the order of design m2 §10.4: refusals (nothing is written),
    the devices, the roles, the labels, the blocks of the previous file, and the
    notes. A refused plan still says what it found, so the report can show the
    devices that did not initialise; an incomplete scan says only that, since its
    rows are the ones a dead child had finished.

    Args:
        scan: What the build child wrote.
        previous: The file being replaced, as :func:`read_previous` read it.
        overrides: The profile's ``[roles.assign]``.
        exclusions: The profile's ``[roles.exclude]``.
        skip: The ``--skip`` names, to say which matched no device.
    """
    stand, camera = scan.stand, scan.camera
    plan = ConfigPlan(
        adapter=stand.adapter,
        camera_adapter=camera.adapter if camera is not None else None,
        devices=[],
        roles=resolve([]),
        carried=[],
        not_carried=[],
        left_out=[],
        refusals=_refusals(scan),
        notes=[],
    )
    if cut_short(scan):
        return plan

    plan.devices, plan.left_out = _select_devices(scan)
    plan.roles = resolve(
        [DeviceInfo(d.label, d.type, d.library, d.name) for d in plan.devices],
        core_roles=previous.core_slots if previous is not None else None,
        overrides=overrides,
        exclusions=exclusions,
    )
    plan.notes += _core_slots(plan.roles, plan.devices)[1]

    # The device's own labels, the ones that can be written (rule 4).
    written = {d.label: d for d in plan.devices}
    own_labels: dict[str, list[str]] = {}
    for device in plan.devices:
        if device.type != "State":
            continue
        usable: list[str] = []
        for position, label in enumerate(device.state_labels):
            why = _unwritable_reason(label)
            if why is None:
                usable.append(label)
            else:
                plan.notes.append(
                    f"{device.label}: the label of position {position} was not "
                    f"written: {why}"
                )
        own_labels[device.label] = usable

    seen = {
        row.name for walk in (stand, camera) if walk is not None for row in walk.devices
    }
    if previous is not None:
        plan.carried, plan.not_carried = _split_previous(
            previous, written, own_labels, seen
        )

    if not any(d.type == "Camera" for d in plan.devices):
        plan.notes.append(
            f"no camera: {stand.adapter} provides none; pass --camera-adapter with "
            "the camera's adapter (smc discover lists the installed ones)"
        )
    for name in skip:
        if name not in seen:
            where = stand.adapter + (
                f" or {camera.adapter}" if camera is not None else ""
            )
            plan.notes.append(f"--skip {name} matched no device of {where}")
    return plan


# --- the text ------------------------------------------------------------------------


def _comment(text: str) -> str:
    """One comment line: runs of whitespace, line breaks included, become one space.

    MMCore's own errors span several lines (design §7.1) and a device name can
    hold one: in a comment the second half would be read as a directive (FM-37's
    lesson, for files). A text that would start the sentinel of a set-aside line
    is put in parentheses, so that no comment is ever read back as a line.
    """
    body = " ".join(text.split())
    line = f"# {body}".rstrip()
    if line.startswith(NOT_CARRIED):
        return f"# ({body})"
    return line


def render_config(plan: ConfigPlan, *, header: Sequence[str]) -> str:
    """The text of the new file, with LF line ends (design m2 §10.4).

    A section with nothing in it is left out, with its heading. Labels come
    before the presets, so that a preset that selects a label finds it defined.

    Raises:
        ValueError: The plan has refusals: nothing may be written from it.
    """
    if not plan.writable:
        raise ValueError(
            "a plan with refusals has no file: " + "; ".join(plan.refusals)
        )
    sections: list[list[str]] = []

    intro = [_comment(line) for line in header]
    intro.append(
        _comment(
            "Only the devices that initialised are loaded"
            + ("; the others are listed at the end." if plan.left_out else ".")
        )
    )
    sections.append(intro)
    sections.append(["# Reset", "Property,Core,Initialize,0"])
    sections.append(
        ["# Devices"] + [f"Device,{d.label},{d.library},{d.name}" for d in plan.devices]
    )
    parents = [f"Parent,{d.label},{d.parent}" for d in plan.devices if d.parent]
    if parents:
        sections.append(["# Hub (parent) references", *parents])
    sections.append(["# Initialize", "Property,Core,Initialize,1"])

    slots, _ = _core_slots(plan.roles, plan.devices)
    if slots:
        roles = ["# Roles"] + [f"Property,Core,{slot},{label}" for slot, label in slots]
        if any(slot == "Shutter" for slot, _ in slots):
            roles.append("Property,Core,AutoShutter,1")
        sections.append(roles)

    from_previous = {
        block.key: block for block in plan.carried if block.kind == "labels"
    }
    labels: list[str] = []
    for device in plan.devices:
        if device.type != "State":
            continue
        if device.label in from_previous:
            labels.append(_comment(f"{device.label} (from the previous configuration)"))
            labels.extend(from_previous[device.label].lines)
            continue
        own = [
            f"Label,{device.label},{position},{label}"
            for position, label in enumerate(device.state_labels)
            if _unwritable_reason(label) is None
        ]
        if own:
            labels.append(_comment(device.label))
            labels.extend(own)
    if labels:
        sections.append(["# Labels", *labels])

    carried = [block for block in plan.carried if block.kind != "labels"]
    if carried:
        body: list[str] = []
        for block in carried:
            body.extend(block.lines)
        sections.append(["# Carried over from the previous configuration", *body])

    if plan.not_carried:
        aside = [
            _comment(
                "Set aside: kept as comments, carried again by a later build once "
                "they apply"
            )
        ]
        for block, why in plan.not_carried:
            aside.append(_comment(f"{block.key}: {why}"))
            aside.extend(f"{NOT_CARRIED}{line}" for line in block.lines)
        sections.append(aside)

    if plan.left_out:
        sections.append(
            ["# Did not connect, or left out"]
            + ["#   " + " ".join(f"{key}: {why}".split()) for key, why in plan.left_out]
        )

    return "\n\n".join("\n".join(section) for section in sections) + "\n"
