"""The NIS-Elements 6.20 multipoint list: ``encode`` and ``decode`` against the real export (m3 §14, #196).

``tests/data/nis/nis-elements-6.20-multipoints.xml`` is the file NIS-Elements 6.20 wrote on the
Ti2-E on 2026-10-29: ten positions, ``bIncludeZ`` true, ``bPFSEnabled`` false, and a
``dPFSOffset`` of 9370 in every point. The expected texts are written here from what the
sample holds, never from ``encode``'s own output (FM-124).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest

from smc.plugins.nis_export.format import SUFFIX, NisPosition, decode, encode

SAMPLE_PATH = (
    Path(__file__).parents[1] / "data" / "nis" / "nis-elements-6.20-multipoints.xml"
)
SAMPLE_SHA256 = "9eebeadfe97518f060e2cbdcfc0ce1fc11fa23253346305f4daa8a8ffcc18d16"
UNUSED_OFFSET = 'dPFSOffset runtype="double" value="9370.000000000000000"'
LEAVES = "bChecked, strName, dXPosition, dYPosition, dZPosition, dPFSOffset, baUserData"


def _exactly(message: str) -> str:
    """A ``match=`` that holds the whole message and nothing else (FM-45)."""
    return f"^{re.escape(message)}$"


def _sample() -> bytes:
    return SAMPLE_PATH.read_bytes()


def _text() -> str:
    return _sample().decode("utf-16")


def _utf16(text: str) -> bytes:
    return text.encode("utf-16")


def _in_point(text: str, point: int, old: str, new: str) -> str:
    """``text`` with ``old`` replaced by ``new`` inside ``Point<point>`` only, found there."""
    start = text.index(f"<Point{point:05d} ")
    end = text.index(f"</Point{point:05d}>") + len(f"</Point{point:05d}>")
    inside = text[start:end]
    assert inside.count(old) == 1, f"{old!r} is not once in Point{point:05d}"
    return text[:start] + inside.replace(old, new) + text[end:]


def _once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"{old!r} is not once in the sample"
    return text.replace(old, new)


def _leaf_of(text: str, point: int, tag: str) -> str:
    """The sample's own ``<tag runtype=... value=.../>`` of ``Point<point>``, as written."""
    start = text.index(f"<Point{point:05d} ")
    end = text.index(f"</Point{point:05d}>")
    found = re.findall(rf'<{tag} runtype="[^"]*" value="[^"]*"/>', text[start:end])
    assert len(found) == 1
    return found[0]


def _with_value(text: str, point: int, tag: str, value: str) -> str:
    """``text`` with the value of ``Point<point>``'s ``tag`` leaf set to ``value``."""
    leaf = _leaf_of(text, point, tag)
    return _in_point(
        text, point, leaf, re.sub(r'value="[^"]*"', f'value="{value}"', leaf)
    )


# ---------------------------------------------------------------- the real sample


def test_the_committed_sample_is_the_nis_export_byte_for_byte() -> None:
    # FM-158: a text hook or a checkout that rewrites it fails here, by name.
    data = _sample()

    assert len(data) == 8846
    assert data.startswith(b"\xff\xfe<\x00")
    assert hashlib.sha256(data).hexdigest() == SAMPLE_SHA256


def test_the_sample_decodes_to_its_ten_points() -> None:
    points = decode(_sample())

    assert len(points) == 10
    assert points[0] == NisPosition("", -8271.0, 4822.0, 4089.12, None)
    assert points[2].x_um == -11613.300000000001
    assert points[2].y_um == 5612.6
    assert points[2].z_um == 4089.1000000000004
    # bPFSEnabled is false, so the 9370 of every point is not an offset.
    assert all(point.pfs_offset is None for point in points)
    assert all(point.name == "" for point in points)


def test_the_sample_re_encodes_to_its_own_bytes_but_the_unused_offset() -> None:
    sample = _text()
    assert sample.count(UNUSED_OFFSET) == 10  # so the replacement below is not vacuous
    expected = sample.replace(
        UNUSED_OFFSET, 'dPFSOffset runtype="double" value="0.000000000000000"'
    )

    again = encode(decode(_sample())).decode("utf-16")

    assert again == expected


def test_the_sample_s_own_values_re_encode_byte_for_byte_but_the_flag() -> None:
    sample = _text()
    flag = 'bPFSEnabled runtype="bool" value="false"'
    assert sample.count(flag) == 1
    points = [dataclasses.replace(p, pfs_offset=9370.0) for p in decode(_sample())]

    again = encode(points).decode("utf-16")

    assert again == sample.replace(flag, 'bPFSEnabled runtype="bool" value="true"')


# ------------------------------------------------------------------------ encode


def _position(
    name: str = "B03_r000",
    x_um: float = -11613.300000000001,
    y_um: float = 12.25,
    z_um: float | None = 4089.12,
    pfs_offset: float | None = 9370.0,
) -> NisPosition:
    return NisPosition(name, x_um, y_um, z_um, pfs_offset)


def test_the_suffix_is_xml() -> None:
    assert SUFFIX == ".xml"


def test_encode_writes_utf16_le_with_a_bom_and_no_line_break() -> None:
    data = encode([_position(), _position("B03_r001")])

    assert data.startswith(
        b"\xff\xfe<\x00?\x00x\x00m\x00l\x00"
    )  # BOM, then little-endian
    text = data.decode("utf-16")
    assert text.startswith(
        '<?xml version="1.0" encoding="UTF-16"?><variant version="1.0">'
        '<no_name runtype="CLxListVariant">'
    )
    assert text.endswith("</no_name></variant>")
    assert "\n" not in text
    assert "\r" not in text
    assert "<Point00000 " in text
    assert "<Point00001 " in text
    assert "<Point00002 " not in text
    assert " />" not in text  # ElementTree's spelling; NIS writes `/>` after the quote


@pytest.mark.parametrize(
    ("z_um", "pfs_offset"),
    [
        pytest.param(4089.12, 9370.0, id="z and offset"),
        pytest.param(4089.12, None, id="z only"),
        pytest.param(None, None, id="neither"),
        pytest.param(None, 9370.0, id="offset without z"),
    ],
)
def test_a_round_trip_keeps_every_value_in_each_mode(
    z_um: float | None, pfs_offset: float | None
) -> None:
    positions = (
        _position("B03_r000", -11613.300000000001, 12.25, z_um, pfs_offset),
        _position("B03_r001", -40000.5, 5612.6, z_um, pfs_offset),
        _position("B04_r000", 12.25, -40000.5, z_um, pfs_offset),
    )

    data = encode(positions)

    assert decode(data) == positions
    text = data.decode("utf-16")
    assert f'bIncludeZ runtype="bool" value="{str(z_um is not None).lower()}"/>' in text
    assert (
        f'bPFSEnabled runtype="bool" value="{str(pfs_offset is not None).lower()}"/>'
        in text
    )


def test_a_field_without_its_flag_is_written_as_zero() -> None:
    text = encode([_position(z_um=None, pfs_offset=None)]).decode("utf-16")

    assert 'dZPosition runtype="double" value="0.000000000000000"/>' in text
    assert 'dPFSOffset runtype="double" value="0.000000000000000"/>' in text


def test_a_value_below_ten_comes_back_within_1e_15_not_exactly() -> None:
    # Fifteen decimals, as NIS writes them: `repr` would round-trip it exactly and so
    # would not be the file NIS wrote. 26 % of the values in [1, 10) go through this.
    value = 1.2345678901234567

    back = decode(encode([_position(x_um=value)]))[0].x_um

    assert back != value
    assert abs(back - value) <= 1e-15


def test_a_value_below_ten_is_written_with_fifteen_decimals() -> None:
    text = encode([_position(x_um=1.5, y_um=-0.25)]).decode("utf-16")

    assert 'dXPosition runtype="double" value="1.500000000000000"/>' in text
    assert 'dYPosition runtype="double" value="-0.250000000000000"/>' in text


def test_a_list_where_only_some_positions_have_z_is_refused() -> None:
    positions = [_position(), _position(z_um=None), _position()]

    with pytest.raises(
        ValueError,
        match=_exactly(
            "2 of 3 positions have z_um and the others do not: one bIncludeZ flag "
            "covers the whole list"
        ),
    ):
        encode(positions)


def test_a_list_where_only_some_positions_have_an_offset_is_refused() -> None:
    positions = [_position(pfs_offset=None), _position(pfs_offset=None), _position()]

    with pytest.raises(
        ValueError,
        match=_exactly(
            "1 of 3 positions have pfs_offset and the others do not: one bPFSEnabled "
            "flag covers the whole list"
        ),
    ):
        encode(positions)


def test_a_name_with_markup_round_trips_inside_double_quotes() -> None:
    name = "a\"&<b>'"

    data = encode([_position(name)])

    assert (
        'strName runtype="CLxStringW" value="a&quot;&amp;&lt;b&gt;\'"/>'
        in data.decode("utf-16")
    )
    assert decode(data)[0].name == name


@pytest.mark.parametrize(
    "ch",
    [
        pytest.param(chr(0x20), id="space"),
        pytest.param(chr(0xD7FF), id="below the surrogates"),
        pytest.param(chr(0xE000), id="above the surrogates"),
        pytest.param(chr(0xFFFD), id="replacement character"),
        pytest.param(chr(0x1F600), id="outside the BMP"),
        pytest.param("\N{GREEK SMALL LETTER MU}", id="mu"),
    ],
)
def test_a_name_xml_can_carry_round_trips(ch: str) -> None:
    name = f"B{ch}1"

    assert decode(encode([_position(name)]))[0].name == name


@pytest.mark.parametrize(
    "ch",
    [
        pytest.param(chr(0x01), id="control"),
        pytest.param(chr(0x09), id="tab"),
        pytest.param(chr(0x1F), id="last control"),
        pytest.param(chr(0xD800), id="first surrogate"),
        pytest.param(chr(0xDFFF), id="last surrogate"),
        pytest.param(chr(0xFFFE), id="FFFE"),
        pytest.param(chr(0xFFFF), id="FFFF"),
    ],
)
def test_a_name_xml_cannot_carry_is_refused(ch: str) -> None:
    name = f"B{ch}1"

    with pytest.raises(
        ValueError,
        match=_exactly(
            f"position 1: name {name!r} holds {ch!r}, which the list cannot carry"
        ),
    ):
        encode([_position(), _position(name)])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=repr)
@pytest.mark.parametrize("field", ["x_um", "y_um", "z_um", "pfs_offset"])
def test_a_value_that_is_not_finite_is_refused(field: str, bad: float) -> None:
    positions = [_position(), dataclasses.replace(_position(), **{field: bad})]

    with pytest.raises(
        ValueError,
        match=_exactly(f"position 1: {field} is {bad!r}, not a finite number"),
    ):
        encode(positions)


def test_an_empty_list_has_both_flags_false_and_decodes_empty() -> None:
    data = encode([])

    text = data.decode("utf-16")
    assert 'bIncludeZ runtype="bool" value="false"/>' in text
    assert 'bPFSEnabled runtype="bool" value="false"/>' in text
    assert "<Point" not in text
    assert decode(data) == ()


def test_more_than_100000_positions_are_refused() -> None:
    with pytest.raises(
        ValueError,
        match=_exactly("100001 positions: the format numbers at most 100000"),
    ):
        encode([_position()] * 100_001)


def test_100000_positions_are_the_most_the_format_numbers() -> None:
    data = encode([_position()] * 100_000)

    text = data.decode("utf-16")
    assert "<Point99999 " in text
    assert "<Point100000" not in text


# ------------------------------------------------------------------------ decode


def test_decode_reads_a_list_that_has_the_flags_set_the_other_way() -> None:
    sample = _once(
        _text(),
        'bPFSEnabled runtype="bool" value="false"',
        'bPFSEnabled runtype="bool" value="true"',
    )
    sample = _once(
        sample,
        'bIncludeZ runtype="bool" value="true"',
        'bIncludeZ runtype="bool" value="false"',
    )

    points = decode(_utf16(sample))

    assert all(point.z_um is None for point in points)
    assert all(point.pfs_offset == 9370.0 for point in points)


def test_decode_refuses_a_parse_error_and_keeps_it_as_its_cause() -> None:
    with pytest.raises(
        ValueError, match=r"^not well-formed XML \(unclosed token"
    ) as info:
        decode(_sample()[:-10])

    assert isinstance(info.value.__cause__, ET.ParseError)


@pytest.mark.parametrize(
    "damaged",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"abc", id="not XML"),
        pytest.param(_sample() + b"\n", id="one byte more (a text hook, FM-158)"),
    ],
)
def test_decode_refuses_bytes_that_are_not_the_export(damaged: bytes) -> None:
    with pytest.raises(ValueError, match=r"^not well-formed XML \(") as info:
        decode(damaged)

    assert isinstance(info.value.__cause__, ET.ParseError)


def _renamed_root(text: str) -> str:
    assert text.count("variant") == 2
    return text.replace("variant", "variantx")


def _second_child_at_the_root(text: str) -> str:
    return _once(text, "</no_name>", "</no_name><extra/>")


def _flags_swapped(text: str) -> str:
    return _once(
        text,
        '<bIncludeZ runtype="bool" value="true"/>'
        '<bPFSEnabled runtype="bool" value="false"/>',
        '<bPFSEnabled runtype="bool" value="false"/>'
        '<bIncludeZ runtype="bool" value="true"/>',
    )


def _point_renamed(text: str) -> str:
    assert text.count("Point00001") == 2  # the opening and the closing tag
    return text.replace("Point00001", "Point00007")


def _leaves_swapped(text: str) -> str:
    x, y = _leaf_of(text, 4, "dXPosition"), _leaf_of(text, 4, "dYPosition")
    assert x + y in text
    return _in_point(text, 4, x + y, y + x)


#: ``(how the sample is damaged, what decode says)``: the sample's own text, edited in place,
#: and the whole message. Each edit is found exactly once in its point, so no case is vacuous.
REFUSED = [
    pytest.param(
        _renamed_root,
        'the root element is <variantx version="1.0">, not <variant version="1.0">',
        id="root renamed",
    ),
    pytest.param(
        lambda t: _once(t, '<variant version="1.0">', '<variant version="2.0">'),
        'the root element is <variant version="2.0">, not <variant version="1.0">',
        id="version changed",
    ),
    pytest.param(
        _second_child_at_the_root,
        "variant: expected one <no_name>, found no_name, extra",
        id="a second element at the root",
    ),
    pytest.param(
        lambda t: _once(t, 'runtype="CLxListVariant"', 'runtype="CLxList"'),
        "no_name: runtype is 'CLxList', not 'CLxListVariant'",
        id="list runtype changed",
    ),
    pytest.param(
        _flags_swapped,
        "no_name: expected <bIncludeZ>, found <bPFSEnabled>",
        id="flags swapped",
    ),
    pytest.param(
        lambda t: _once(
            t,
            'bIncludeZ runtype="bool" value="true"',
            'bIncludeZ runtype="bool" value="yes"',
        ),
        "no_name/bIncludeZ: 'yes' is not 'true' or 'false'",
        id="a bool that is yes",
    ),
    pytest.param(
        _point_renamed,
        "no_name: expected <Point00001>, found <Point00007>",
        id="points out of sequence",
    ),
    pytest.param(
        lambda t: _once(
            t,
            '<Point00000 runtype="NDSetupMultipointListItem">',
            '<Point00000 runtype="NDSetupMultipointListItems">',
        ),
        "Point00000: runtype is 'NDSetupMultipointListItems', not 'NDSetupMultipointListItem'",
        id="point runtype changed",
    ),
    pytest.param(
        lambda t: _in_point(
            t,
            3,
            'bChecked runtype="bool" value="true"',
            'bChecked runtype="bool" value="false"',
        ),
        "Point00003/bChecked: an unchecked point is not supported",
        id="an unchecked point",
    ),
    pytest.param(
        lambda t: _in_point(
            t, 4, '<dPFSOffset runtype="double" value="9370.000000000000000"/>', ""
        ),
        f"Point00004: expected 7 elements {LEAVES}; found 6: "
        "bChecked, strName, dXPosition, dYPosition, dZPosition, baUserData",
        id="a leaf removed",
    ),
    pytest.param(
        lambda t: _in_point(
            t, 2, "</Point00002>", '<dExtra runtype="double" value="1.0"/></Point00002>'
        ),
        f"Point00002: expected 7 elements {LEAVES}; found 8: {LEAVES}, dExtra",
        id="an extra leaf",
    ),
    pytest.param(
        _leaves_swapped,
        "Point00004: expected <dXPosition>, found <dYPosition>",
        id="two leaves in the other order",
    ),
    pytest.param(
        lambda t: _in_point(
            t, 6, 'dYPosition runtype="double"', 'dYPosition runtype="float"'
        ),
        "Point00006/dYPosition: runtype is 'float', not 'double'",
        id="a leaf runtype changed",
    ),
    pytest.param(
        lambda t: _with_value(t, 1, "dXPosition", "abc"),
        "Point00001/dXPosition: 'abc' is not a finite number",
        id="a double that is not a number",
    ),
    pytest.param(
        lambda t: _with_value(t, 8, "dZPosition", "inf"),
        "Point00008/dZPosition: 'inf' is not a finite number",
        id="a double that is inf",
    ),
    pytest.param(
        lambda t: _with_value(t, 7, "dPFSOffset", "nan"),
        "Point00007/dPFSOffset: 'nan' is not a finite number",
        id="a double that is nan",
    ),
    pytest.param(
        lambda t: _with_value(t, 9, "dYPosition", "1e999"),
        "Point00009/dYPosition: '1e999' is not a finite number",
        id="a double that overflows to inf",
    ),
    pytest.param(
        lambda t: _in_point(
            t,
            5,
            'baUserData runtype="CLxByteArray" value=""',
            'baUserData runtype="CLxByteArray" value="AA=="',
        ),
        "Point00005/baUserData: value 'AA==' is not empty",
        id="user data",
    ),
    pytest.param(
        lambda t: _in_point(
            t,
            0,
            'bChecked runtype="bool" value="true"',
            'bChecked runtype="bool" value="true" x="1"',
        ),
        "Point00000/bChecked: expected the attributes runtype and value only, found ['runtype', 'value', 'x']",
        id="a leaf attribute",
    ),
    pytest.param(
        lambda t: _in_point(
            t,
            0,
            '<bChecked runtype="bool" value="true"/>',
            '<bChecked runtype="bool" value="true"><x/></bChecked>',
        ),
        "Point00000/bChecked: a leaf holds no element, found <x>",
        id="a leaf that holds an element",
    ),
]


@pytest.mark.parametrize(("damage", "message"), REFUSED)
def test_decode_refuses_what_is_not_the_sample_s_structure(
    damage: Callable[[str], str], message: str
) -> None:
    damaged = damage(_text())
    assert damaged != _text()

    with pytest.raises(ValueError, match=_exactly(message)):
        decode(_utf16(damaged))
