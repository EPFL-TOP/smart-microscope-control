"""The XY position list (multipoint) that NIS-Elements 6.20 loads, written and read back (m3 §14).

The format is not documented by Nikon. Everything here was measured on one file,
exported from NIS-Elements 6.20 on the Ti2-E on 2026-10-29 (ten positions, kept as
``tests/data/nis/nis-elements-6.20-multipoints.xml``), and ``encode`` reproduces that
file byte for byte from its own values. What the file shows, and what the code does
with it:

* **UTF-16 with a byte-order mark, and no line break anywhere**, the end included. The
  text is built by hand and encoded with ``str.encode("utf-16")``: ``ElementTree``
  would write `` />`` after each leaf and its own declaration, and NIS wrote neither.
* The list is ``<variant version="1.0">`` holding a ``CLxListVariant`` with two flags,
  ``bIncludeZ`` and ``bPFSEnabled``, then one ``PointNNNNN`` (``NDSetupMultipointListItem``)
  per position. A point holds, in this order, ``bChecked``, ``strName``, ``dXPosition``,
  ``dYPosition``, ``dZPosition``, ``dPFSOffset`` and ``baUserData``, each a leaf
  ``<tag runtype="..." value="..."/>``.
* **Doubles carry fifteen decimals** (``f"{v:.15f}"``), as NIS writes them. A value of
  ten or more survives the round trip exactly; below ten it can come back up to 1e-15
  away, which is far under what a stage resolves.
* **One flag covers the whole list.** A list where some positions have a Z and others
  do not cannot be written, and a field whose flag is false is written as ``0``: it is
  not read back (``decode`` gives ``None``), whatever the file holds. The sample's own
  ``dPFSOffset`` is 9370 in every point while ``bPFSEnabled`` is false, which is why.
* ``bChecked`` is always true and ``baUserData`` always empty; the sample has nothing
  else, and a list that has is refused by ``decode`` rather than half read.

``encode`` refuses what the file cannot carry instead of writing something NIS may read
differently: more than 100 000 points (the numbering has five digits), a value that is not
a finite number, and a name with a character XML 1.0 cannot hold (below U+0020, a lone
surrogate, U+FFFE, U+FFFF). A lone surrogate is refused, not replaced (FM-05): route names
are ``<well>_r<k>`` in ASCII, so meeting one means the record was edited.

``decode`` exists for the round trip and the tests. It is strict, so that a list that
differs from the sample's structure fails with the element that differs, and it is
never used to read a list back into smc.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["SUFFIX", "NisPosition", "decode", "encode"]

SUFFIX = ".xml"

_MAX_POSITIONS = 100_000
_POINT_RUNTYPE = "NDSetupMultipointListItem"
_LEAVES = (
    ("bChecked", "bool"),
    ("strName", "CLxStringW"),
    ("dXPosition", "double"),
    ("dYPosition", "double"),
    ("dZPosition", "double"),
    ("dPFSOffset", "double"),
    ("baUserData", "CLxByteArray"),
)
_HEAD = (
    '<?xml version="1.0" encoding="UTF-16"?><variant version="1.0">'
    '<no_name runtype="CLxListVariant">'
)
_TAIL = "</no_name></variant>"


@dataclass(frozen=True, slots=True)
class NisPosition:
    """One position of the list: a name, X and Y in µm, and the focus when the list carries it.

    ``z_um`` is the stage's Z in µm and ``pfs_offset`` the autofocus offset in
    PFSOffset's own unit, never converted. ``None`` means the list holds none: an
    unknown is not a zero.
    """

    name: str
    x_um: float
    y_um: float
    z_um: float | None
    pfs_offset: float | None


def encode(positions: Sequence[NisPosition]) -> bytes:
    """The positions as the bytes of a multipoint list that NIS-Elements 6.20 loads.

    Raises:
        ValueError: More than 100 000 positions; a value that is not a finite
            number; a name holding a character the list cannot carry; or some
            positions with ``z_um`` (or ``pfs_offset``) and some without, because
            one flag covers the whole list. A message about one position starts
            with ``position <index>``.
    """
    count = len(positions)
    if count > _MAX_POSITIONS:
        raise ValueError(
            f"{count} positions: the format numbers at most {_MAX_POSITIONS}"
        )
    for index, position in enumerate(positions):
        _require_carriable(index, position)
    include_z = _all_or_none([p.z_um for p in positions], "z_um", "bIncludeZ")
    pfs_enabled = _all_or_none(
        [p.pfs_offset for p in positions], "pfs_offset", "bPFSEnabled"
    )
    parts = [
        _HEAD,
        _leaf("bIncludeZ", "bool", _bool_text(include_z)),
        _leaf("bPFSEnabled", "bool", _bool_text(pfs_enabled)),
        *(_point(index, position) for index, position in enumerate(positions)),
        _TAIL,
    ]
    return "".join(parts).encode("utf-16")


def decode(data: bytes) -> tuple[NisPosition, ...]:
    """The positions of a multipoint list, read as strictly as the sample is written.

    A field whose flag (``bIncludeZ``, ``bPFSEnabled``) is false comes back as
    ``None``, whatever value the file holds for it.

    Raises:
        ValueError: The bytes are not well-formed XML (the parse error is the
            ``__cause__``), or the structure is not the sample's: the message names
            the element that differs.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed XML ({exc})") from exc
    if root.tag != "variant" or root.attrib != {"version": "1.0"}:
        attributes = "".join(
            f' {name}="{value}"' for name, value in root.attrib.items()
        )
        raise ValueError(
            f'the root element is <{root.tag}{attributes}>, not <variant version="1.0">'
        )
    if [child.tag for child in root] != ["no_name"]:
        raise ValueError(f"variant: expected one <no_name>, found {_tags(root)}")
    listing = root[0]
    _require_runtype(listing, "no_name", "CLxListVariant")
    items = list(listing)
    if len(items) < 2:
        raise ValueError(
            f"no_name: expected <bIncludeZ> and <bPFSEnabled> first, found {_tags(listing)}"
        )
    include_z = _as_bool(
        _leaf_value(items[0], "bIncludeZ", "bool", "no_name"), "no_name/bIncludeZ"
    )
    pfs_enabled = _as_bool(
        _leaf_value(items[1], "bPFSEnabled", "bool", "no_name"), "no_name/bPFSEnabled"
    )
    return tuple(
        _decode_point(index, element, include_z, pfs_enabled)
        for index, element in enumerate(items[2:])
    )


# ---------------------------------------------------------------------- writing


def _require_carriable(index: int, position: NisPosition) -> None:
    """Refuse the first thing in ``position`` that the file cannot carry."""
    for char in position.name:
        code = ord(char)
        if code < 0x20 or 0xD800 <= code <= 0xDFFF or code in (0xFFFE, 0xFFFF):
            raise ValueError(
                f"position {index}: name {position.name!r} holds {char!r}, "
                f"which the list cannot carry"
            )
    for field, value in (
        ("x_um", position.x_um),
        ("y_um", position.y_um),
        ("z_um", position.z_um),
        ("pfs_offset", position.pfs_offset),
    ):
        if value is not None and not math.isfinite(value):
            raise ValueError(
                f"position {index}: {field} is {value!r}, not a finite number"
            )


def _all_or_none(values: Sequence[float | None], field: str, flag: str) -> bool:
    """Whether every position has ``field``; a mixed list has no flag to write (``ValueError``)."""
    given = sum(value is not None for value in values)
    if given not in (0, len(values)):
        raise ValueError(
            f"{given} of {len(values)} positions have {field} and the others do not: "
            f"one {flag} flag covers the whole list"
        )
    return given > 0


def _point(index: int, position: NisPosition) -> str:
    tag = f"Point{index:05d}"
    z_um = 0.0 if position.z_um is None else position.z_um
    pfs_offset = 0.0 if position.pfs_offset is None else position.pfs_offset
    values = (
        "true",
        # Double quotes inside double quotes: `quoteattr` would switch to single ones.
        escape(position.name, {'"': "&quot;"}),
        _number(position.x_um),
        _number(position.y_um),
        _number(z_um),
        _number(pfs_offset),
        "",
    )
    leaves = "".join(
        _leaf(name, runtype, value)
        for (name, runtype), value in zip(_LEAVES, values, strict=True)
    )
    return f'<{tag} runtype="{_POINT_RUNTYPE}">{leaves}</{tag}>'


def _leaf(tag: str, runtype: str, value: str) -> str:
    return f'<{tag} runtype="{runtype}" value="{value}"/>'


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _number(value: float) -> str:
    return f"{value:.15f}"


# ---------------------------------------------------------------------- reading


def _tags(element: ET.Element) -> str:
    return ", ".join(child.tag for child in element) or "nothing"


def _require_runtype(element: ET.Element, path: str, runtype: str) -> None:
    found = element.get("runtype")
    if found != runtype:
        raise ValueError(f"{path}: runtype is {found!r}, not {runtype!r}")


def _leaf_value(element: ET.Element, tag: str, runtype: str, parent: str) -> str:
    """The ``value`` of ``element``, which must be the leaf ``<tag runtype=... value=.../>``."""
    path = f"{parent}/{tag}"
    if element.tag != tag:
        raise ValueError(f"{parent}: expected <{tag}>, found <{element.tag}>")
    _require_runtype(element, path, runtype)
    if sorted(element.attrib) != ["runtype", "value"]:
        raise ValueError(
            f"{path}: expected the attributes runtype and value only, "
            f"found {sorted(element.attrib)}"
        )
    if len(element):
        raise ValueError(f"{path}: a leaf holds no element, found <{element[0].tag}>")
    return element.attrib["value"]


def _as_bool(value: str, path: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError(f"{path}: {value!r} is not 'true' or 'false'")
    return value == "true"


def _as_double(value: str, path: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(f"{path}: {value!r} is not a finite number")
    return number


def _decode_point(
    index: int, element: ET.Element, include_z: bool, pfs_enabled: bool
) -> NisPosition:
    path = f"Point{index:05d}"
    if element.tag != path:
        raise ValueError(f"no_name: expected <{path}>, found <{element.tag}>")
    _require_runtype(element, path, _POINT_RUNTYPE)
    if len(element) != len(_LEAVES):
        raise ValueError(
            f"{path}: expected {len(_LEAVES)} elements "
            f"{', '.join(name for name, _ in _LEAVES)}; "
            f"found {len(element)}: {_tags(element)}"
        )
    checked, name, x, y, z, pfs, user_data = (
        _leaf_value(child, tag, runtype, path)
        for child, (tag, runtype) in zip(element, _LEAVES, strict=True)
    )
    if not _as_bool(checked, f"{path}/bChecked"):
        raise ValueError(f"{path}/bChecked: an unchecked point is not supported")
    x_um = _as_double(x, f"{path}/dXPosition")
    y_um = _as_double(y, f"{path}/dYPosition")
    z_um = _as_double(z, f"{path}/dZPosition")
    pfs_offset = _as_double(pfs, f"{path}/dPFSOffset")
    if user_data:
        raise ValueError(f"{path}/baUserData: value {user_data!r} is not empty")
    return NisPosition(
        name,
        x_um,
        y_um,
        z_um if include_z else None,
        pfs_offset if pfs_enabled else None,
    )
