"""Channel presets: the pure half of ``Channels`` (m2 §8.5, ADR-0004).

A channel is a Micro-Manager config-group preset: which filters, which light
path, how bright. This module decides what needs no device: which group holds
the channels, which presets the layer refuses to apply, whether a value can
be written to a ``.cfg``, whether a setting matches what the stand reads, and
the text edit that replaces one preset in a configuration file. It imports no
Micro-Manager code, so every rule here is tested without a core.

Hardware behaviour encoded here (measured on the demo, 2026-10-19, m2 §8.1):

* **A group is chosen by its name**, never by what it holds: the demo and the
  Ti2-E both call theirs ``Channel``, and the demo's ``Camera``,
  ``Objective`` and ``System`` groups would change binning, turn the
  objective or re-run the start-up preset.
* **A preset can move what the layer guards.** One ``setConfig`` can turn the
  objective turret, move a stage or switch the autofocus, around every guard
  of those capabilities; such a preset is refused, and cannot be forced.
* **MMCore compares presets as strings**: ``Exposure`` set from ``10`` reads
  back ``10.0000``, and ``getCurrentConfig`` then names no preset. Numbers are
  compared as numbers here.
* **The ``.cfg`` is comma-separated text** (FM-81): a value with a comma, or
  bytes an adapter returned that are not ASCII (FM-05), would corrupt the
  line it is written into.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from smc.hardware.errors import HardwareError

__all__ = [
    "ALLOWED_CORE_SETTINGS",
    "CHANNEL_WORD",
    "FORBIDDEN_TYPES",
    "NOT_CHANNEL_WORDS",
    "check_preset_name",
    "check_value",
    "choose_group",
    "forbidden_settings",
    "named_for_channels",
    "replace_preset",
    "setting_matches",
]

#: What a group's normalised name contains when it holds channels.
CHANNEL_WORD = "channel"
#: What it never contains: ``Camera-Channel`` sets a camera, not a channel.
NOT_CHANNEL_WORDS = ("camera", "objective", "system")
#: The ``Core`` settings a channel may change: which shutter opens, and
#: whether the core opens it. Every other one moves a core slot.
ALLOWED_CORE_SETTINGS = ("Shutter", "AutoShutter")
#: Device types a channel never sets: each moves only through its capability.
FORBIDDEN_TYPES = ("Stage", "XYStage", "AutoFocus")

_CORE = "Core"
#: The capability that moves each forbidden type, for the refusal's advice.
_CAPABILITY_OF = {"Stage": "ZStage", "XYStage": "XYStage", "AutoFocus": "Autofocus"}


def _norm(text: str) -> str:
    """Lower-case, ``[a-z0-9]`` only, as role names are matched (design §4)."""
    return "".join(c for c in text.lower() if c.isascii() and c.isalnum())


def _printable_ascii(text: str) -> bool:
    return all(" " <= c <= "~" for c in text)


def named_for_channels(group: str) -> bool:
    """Whether ``group``'s name says it holds channels: ``channel``, and no camera, objective or system."""
    name = _norm(group)
    return CHANNEL_WORD in name and not any(word in name for word in NOT_CHANNEL_WORDS)


def choose_group(
    groups: Sequence[str], *, core_group: str, profile_group: str
) -> tuple[str | None, list[str]]:
    """The group whose presets are channels, and the warnings to log; ``None``: no group.

    The profile's ``[channels] group`` wins, whether the group exists yet or
    not (``capture()`` creates it). Then the core's channel group, when it
    exists and is named for channels. Then the one existing group named for
    channels: with several, none is guessed, and a warning names them.
    """
    if profile_group:
        return profile_group, []
    if core_group in groups and named_for_channels(core_group):
        return core_group, []
    named = [group for group in groups if named_for_channels(group)]
    if len(named) == 1:
        return named[0], []
    if not named:
        return None, []
    listed = ", ".join(repr(group) for group in named)
    return None, [
        f"channels: several config groups are named for channels ({listed}) and "
        f"the core's channel group ({core_group!r}) is none of them, so none is "
        f"chosen; name one in the profile's [channels] group"
    ]


def check_preset_name(name: str) -> None:
    """Refuse a preset name that the ``.cfg`` could not hold as written (FM-81).

    Raises:
        ValueError: ``name`` is empty, holds a comma or a character that is not
            printable ASCII, or starts or ends with whitespace.
    """
    problem = ""
    if not name:
        problem = "it is empty"
    elif "," in name:
        problem = "it contains a comma, which separates the fields of a .cfg line"
    elif not _printable_ascii(name):
        problem = "it contains a character that is not printable ASCII"
    elif name != name.strip():
        problem = "it starts or ends with whitespace, which MMCore does not keep"
    if problem:
        raise ValueError(f"channel preset name {name!r} is refused: {problem}")


def forbidden_settings(
    settings: Iterable[tuple[str, str, str]],
    *,
    device_types: Mapping[str, str],
    objective_turret: str | None,
) -> list[str]:
    """Why each setting of a preset may not be applied; empty: the preset is allowed.

    ``device_types`` maps each loaded device's label to its type name
    (``"State"``, ``"Stage"``); a device it lacks is not loaded, and MMCore's
    ``defineConfig`` accepts one without a word (m2 §8.1).
    """
    reasons: list[str] = []
    for device, prop, _value in settings:
        setting = f"{device},{prop}"
        if device == _CORE:
            if prop not in ALLOWED_CORE_SETTINGS:
                reasons.append(f"{setting} changes a core slot")
            continue
        kind = device_types.get(device)
        if kind is None:
            reasons.append(f"{setting}: {device!r} is not loaded")
        elif objective_turret is not None and device == objective_turret:
            reasons.append(
                f"{setting}: {device!r} is the objective turret (use "
                f"ObjectiveTurret.select with confirm=True)"
            )
        elif kind in FORBIDDEN_TYPES:
            reasons.append(
                f"{setting}: {device!r} is a {kind} device (use {_CAPABILITY_OF[kind]})"
            )
    return reasons


def check_value(device: str, prop: str, value: str) -> None:
    """Refuse a value that a ``.cfg`` line cannot hold (FM-81, FM-05).

    Raises:
        HardwareError: ``value`` is empty, holds a comma, or holds a character
            that is not printable ASCII (an adapter's bytes may not even be
            UTF-8).
    """
    if value and "," not in value and _printable_ascii(value):
        return
    # The value may hold a lone surrogate (FM-05): repr() escapes it.
    raise HardwareError(
        f"{device},{prop} reads {value!r}, which a configuration file cannot hold "
        f"(it is empty, contains a comma, or a character that is not printable "
        f"ASCII); nothing was written. Rename that label in the configuration, "
        f"or list {device!r} under [channels] exclude in the profile"
    )


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def setting_matches(expected: str, read: str) -> bool:
    """Whether a preset's ``expected`` value is what the stand ``read``.

    Equal strings match, and so do two numbers that are equal as floats:
    MMCore formats a number it stores (``10`` reads ``10.0000``). A value that
    reads as a number that is not finite (``nan``, ``inf``) never matches:
    ``nan`` equals nothing, and a stand that reports it reports no setting.
    """
    a, b = _number(expected), _number(read)
    if any(n is not None and not math.isfinite(n) for n in (a, b)):
        return False
    if a is not None and b is not None:
        return a == b
    return expected == read


def _fields(line: str) -> list[str]:
    return [field.strip() for field in line.rstrip("\r").split(",")]


def replace_preset(
    text: str, group: str, name: str, settings: Sequence[tuple[str, str, str]]
) -> str:
    """``text``, a ``.cfg``, with preset ``name`` of ``group`` replaced by ``settings``.

    A line belongs to the preset when its first three comma-separated fields,
    stripped, are ``ConfigGroup``, ``group`` and ``name``. Those lines are
    removed, and the new ones go where the first of them was; with none,
    after the group's last line; with no line of the group, at the end. Every
    other line is kept byte for byte, comments included, so that a stand's
    hand-written configuration survives (``saveSystemConfiguration`` would
    rewrite all of it). The new lines end as the file does, CRLF when it holds
    one (a Windows ``.cfg``), else LF, and the text ends with a line ending.
    """
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    else:
        # The last line had no ending: it gets the file's, as the new text ends with one.
        lines[-1] += eol[:-1]
    new = [f"ConfigGroup,{group},{name},{d},{p},{v}" + eol[:-1] for d, p, v in settings]
    own = [
        i
        for i, line in enumerate(lines)
        if _fields(line)[:3] == ["ConfigGroup", group, name]
    ]
    if own:
        at, removed = own[0], set(own)
        kept = [line for i, line in enumerate(lines) if i not in removed]
        lines = [*kept[:at], *new, *kept[at:]]
    else:
        in_group = [
            i
            for i, line in enumerate(lines)
            if _fields(line)[:2] == ["ConfigGroup", group]
        ]
        at = in_group[-1] + 1 if in_group else len(lines)
        lines = [*lines[:at], *new, *lines[at:]]
    return "".join(line + "\n" for line in lines)
