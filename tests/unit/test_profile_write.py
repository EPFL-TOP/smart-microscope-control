"""The ``[objectives]`` block that ``smc objective offset --write`` puts into a profile (m3 §20.7, #380).

A profile is a file people maintain by hand and keep in git (FM-107): the writer
touches the lines between two marker lines and none other. The tests run on the
bytes of ``profiles/nikon-ti2.toml``, the profile this is for, and prove that
every line outside the block comes back as it was, with its line ends (a profile
saved by Notepad has CRLF), and that a byte-order mark is dropped, which the
loader reads the same (FM-80).

The numbers are written here by hand. What is written is *read back* the way the
loader reads it, so a text that would load as other offsets is refused before
anything is written (FM-156).
"""

from __future__ import annotations

import codecs
import sys
from pathlib import Path

import pytest

from smc.hardware import profile_write
from smc.hardware.errors import ProfileError
from smc.hardware.profile import (
    ObjectiveOffsetEntry,
    ObjectivesSection,
    Profile,
    profile_text,
)
from smc.hardware.profile_write import (
    BLOCK_BEGIN,
    BLOCK_END,
    objectives_block,
    with_objectives,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

NIKON = Path(__file__).resolve().parents[2] / "profiles" / "nikon-ti2.toml"
WHERE = "profiles/nikon-ti2.toml"
MEASURED = "2026-11-09, session 3 (#400), objective-offsets in run.json"

SECTION = ObjectivesSection(
    reference="2-Plan Fluor 10x",
    offsets={
        "1-Plan Fluor 4x": ObjectiveOffsetEntry(
            xy_offset_um=(-5.0, 2.5), measured=MEASURED
        ),
        "3-Plan Apo LmbdD0.25 40x": ObjectiveOffsetEntry(
            xy_offset_um=(-37.49, 12.25), measured=MEASURED
        ),
    },
)
OTHER = ObjectivesSection(
    reference="1-Plan Fluor 4x",
    offsets={
        "2-Plan Fluor 10x": ObjectiveOffsetEntry(
            xy_offset_um=(5.0, -2.5), measured="2026-11-12, session 4 (#401)"
        )
    },
)


@pytest.fixture(scope="module")
def original() -> str:
    """The text of ``nikon-ti2.toml``, read as bytes and decoded as the loader does."""
    text = NIKON.read_bytes().decode("utf-8")
    assert text.endswith("\n")
    assert "objectives" not in tomllib.loads(text)
    return text


def _objectives_of(text: str) -> ObjectivesSection:
    return Profile.model_validate(tomllib.loads(text)).objectives


def _without_objectives(text: str) -> dict[str, object]:
    data = tomllib.loads(text)
    data.pop("objectives", None)
    return data


# ---------------------------------------------------------------------- the block


def test_the_block_is_marked_and_says_a_write_replaces_it() -> None:
    block = objectives_block(SECTION)

    lines = block.split("\n")
    assert lines[0] == BLOCK_BEGIN
    assert lines[1] == (
        "# The next write replaces everything up to the closing marker; edit values "
        "here only if no write follows."
    )
    assert lines[-2:] == [BLOCK_END, ""]
    assert BLOCK_BEGIN.startswith("# >>> [objectives], written by smc objective offset")
    assert BLOCK_END.startswith("# <<< [objectives], written by smc objective offset")
    # Between the comment and the closing marker is the TOML of the section.
    body = "\n".join(lines[2:-2])
    assert tomllib.loads(body) == {
        "objectives": {
            "reference": "2-Plan Fluor 10x",
            "offsets": {
                "1-Plan Fluor 4x": {"xy_offset_um": [-5.0, 2.5], "measured": MEASURED},
                "3-Plan Apo LmbdD0.25 40x": {
                    "xy_offset_um": [-37.49, 12.25],
                    "measured": MEASURED,
                },
            },
        }
    }


def test_every_line_of_the_block_ends_with_the_given_line_end() -> None:
    block = objectives_block(SECTION, newline="\r\n")

    assert block.endswith(BLOCK_END + "\r\n")
    # No line ends with a bare LF: removing every CRLF leaves no LF.
    assert "\n" not in block.replace("\r\n", "")
    assert tomllib.loads(block.split("\r\n", 2)[2].rsplit(BLOCK_END, 1)[0])


def test_a_none_value_is_left_out_and_a_z_offset_is_kept() -> None:
    section = ObjectivesSection(
        reference="R",
        offsets={
            "xy only": ObjectiveOffsetEntry(xy_offset_um=(1.0, 2.0), measured=MEASURED),
            "z only": ObjectiveOffsetEntry(z_offset_um=-3.5, measured=MEASURED),
        },
    )

    data = tomllib.loads(objectives_block(section))

    assert data["objectives"]["offsets"] == {
        "xy only": {"xy_offset_um": [1.0, 2.0], "measured": MEASURED},
        "z only": {"z_offset_um": -3.5, "measured": MEASURED},
    }


def test_a_section_with_a_reference_and_no_offsets_is_written() -> None:
    text = objectives_block(ObjectivesSection(reference="R"))

    assert tomllib.loads(text) == {"objectives": {"reference": "R"}}


# ----------------------------------------------------------------------- the write


def test_the_block_is_appended_and_every_other_line_comes_back(original: str) -> None:
    written = with_objectives(original, SECTION, where=WHERE)

    # The bytes of the profile come first, unchanged (FM-107): nothing outside the
    # block is touched, not a line end, not a comment, not a key.
    assert written.encode("utf-8").startswith(NIKON.read_bytes())
    # Then one empty line and the block.
    assert written[len(original) :] == "\n" + objectives_block(SECTION)
    # Parsed, it is the profile it was plus the offsets, and the offsets read back.
    assert _without_objectives(written) == _without_objectives(original)
    assert _objectives_of(written) == SECTION


def test_a_second_write_replaces_the_block_in_place(original: str) -> None:
    first = with_objectives(original, SECTION, where=WHERE)
    after = (
        "# kept: a line after the block\n"
        "[suggest.scan-plan]\n"
        'params = ["coverage=0.4"]\n'
        'why = "a table the writer leaves alone"\n'
    )

    second = with_objectives(first + after, OTHER, where=WHERE)

    assert second == original + "\n" + objectives_block(OTHER) + after
    assert second.count(BLOCK_BEGIN) == 1
    assert second.count(BLOCK_END) == 1
    assert _objectives_of(second) == OTHER


def test_a_block_in_the_middle_is_replaced_and_the_lines_around_it_stay(
    original: str,
) -> None:
    before, rest = original.split("[quirks]", 1)
    text = before + objectives_block(SECTION) + "[quirks]" + rest

    written = with_objectives(text, OTHER, where=WHERE)

    assert written == before + objectives_block(OTHER) + "[quirks]" + rest
    assert _objectives_of(written) == OTHER


def test_crlf_line_ends_are_kept(original: str) -> None:
    crlf = original.replace("\n", "\r\n")

    first = with_objectives(crlf, SECTION, where=WHERE)
    second = with_objectives(first, OTHER, where=WHERE)

    for written in (first, second):
        assert written.startswith(crlf)
        # No line ends with a bare LF, in the profile's lines or in the block's.
        assert "\n" not in written.replace("\r\n", "")
        assert _without_objectives(written) == _without_objectives(original)
    assert first[len(crlf) :] == "\r\n" + objectives_block(SECTION, newline="\r\n")
    assert second == crlf + "\r\n" + objectives_block(OTHER, newline="\r\n")
    assert _objectives_of(second) == OTHER


def test_a_profile_with_a_byte_order_mark_is_written_without_it(
    original: str, tmp_path: Path
) -> None:
    path = tmp_path / "bom.toml"
    path.write_bytes(codecs.BOM_UTF8 + original.encode("utf-8"))

    text = profile_text(path)
    written = with_objectives(text, SECTION, where=str(path)).encode("utf-8")

    assert text == original
    assert not written.startswith(codecs.BOM_UTF8)
    assert written == original.encode("utf-8") + b"\n" + objectives_block(
        SECTION
    ).encode("utf-8")
    # The loader reads it the same, mark or none.
    assert _objectives_of(written.decode("utf-8-sig")) == SECTION


def test_a_final_line_end_is_added_before_the_block() -> None:
    text = '[microscope]\nname = "x"'

    written = with_objectives(text, SECTION, where=WHERE)

    assert written == text + "\n\n" + objectives_block(SECTION)
    assert _objectives_of(written) == SECTION


# ------------------------------------------------------------------- the refusals


@pytest.mark.parametrize(
    "extra",
    [
        '\n[objectives]\nreference = "x"\n',
        'objectives.reference = "x"\n',
        '\n[objectives.offsets."a"]\nmeasured = "m"\n',
    ],
    ids=["table", "dotted-key-at-the-root", "an-entry"],
)
def test_an_objectives_table_outside_the_block_is_refused(
    original: str, extra: str
) -> None:
    # A dotted key lives at the root, so it goes before the first table.
    text = extra + original if extra.startswith("objectives.") else original + extra

    with pytest.raises(ProfileError) as raised:
        with_objectives(text, SECTION, where=WHERE)

    assert str(raised.value) == (
        f"{WHERE}: the profile has an [objectives] table outside the block smc "
        "writes: delete it (its values are printed above) or edit it by hand "
        "instead of writing"
    )


def test_an_objectives_table_outside_a_block_that_exists_is_refused_too(
    original: str,
) -> None:
    # The block is the one smc wrote; the table after it is a hand-written second one.
    text = (
        original
        + objectives_block(SECTION)
        + '[objectives.offsets."by hand"]\nmeasured = "m"\n'
    )

    with pytest.raises(ProfileError, match="outside the block smc writes"):
        with_objectives(text, OTHER, where=WHERE)


@pytest.mark.parametrize(
    ("markers", "opening", "closing"),
    [
        ([BLOCK_BEGIN], 1, 0),
        ([BLOCK_END], 0, 1),
        ([BLOCK_BEGIN, BLOCK_BEGIN, BLOCK_END], 2, 1),
        ([BLOCK_BEGIN, BLOCK_END, BLOCK_END], 1, 2),
        ([BLOCK_END, BLOCK_BEGIN], 1, 1),
        ([BLOCK_BEGIN, BLOCK_END, BLOCK_BEGIN, BLOCK_END], 2, 2),
    ],
    ids=[
        "one-opening",
        "one-closing",
        "two-openings",
        "two-closings",
        "closing-first",
        "two-blocks",
    ],
)
def test_a_damaged_block_is_refused(
    original: str, markers: list[str], opening: int, closing: int
) -> None:
    text = original + "\n".join(markers) + "\n"

    with pytest.raises(ProfileError) as raised:
        with_objectives(text, SECTION, where=WHERE)

    assert str(raised.value) == (
        f"{WHERE}: the [objectives] block that smc writes is damaged ({opening} "
        f"opening and {closing} closing marker lines): restore it from git, or "
        "delete both markers and what lies between, then write again"
    )


def test_a_marker_inside_a_line_is_not_a_marker(original: str) -> None:
    # The lines that *equal* a marker are found: a comment that quotes one is not.
    text = original + f"# see {BLOCK_BEGIN} for what it writes\n"

    written = with_objectives(text, SECTION, where=WHERE)

    assert written.startswith(text)
    assert _objectives_of(written) == SECTION


@pytest.mark.parametrize("trail", [" ", "\t"], ids=["space", "tab"])
def test_a_marker_with_trailing_blanks_is_not_one_and_the_write_appends(
    original: str, trail: str
) -> None:
    # Only an exact line is the writer's own: a marker someone edited is a line of
    # the profile, and is never replaced.
    text = original + BLOCK_BEGIN + trail + "\n"

    written = with_objectives(text, SECTION, where=WHERE)

    assert written.startswith(text)
    # The edited line, and the block that was appended after it.
    assert written.count(BLOCK_BEGIN) == 2
    assert _objectives_of(written) == SECTION


def test_a_profile_that_is_not_toml_outside_the_block_is_refused(original: str) -> None:
    text = original + "this is not toml\n"

    with pytest.raises(ProfileError) as raised:
        with_objectives(text, SECTION, where=WHERE)

    assert str(raised.value).startswith(f"{WHERE}: ")
    assert "not valid TOML" in str(raised.value)


# ------------------------------------------------------------------- the read-back


@pytest.mark.parametrize(
    "label",
    [
        'Nikon "40X" Plan',
        "back\\slash",
        "tab\there",
        "# not a comment [not a table]",
        "micro \N{MICRO SIGN}m",
        "line" + chr(0x2028) + "separator",
        "dotted.name",
    ],
    ids=["quotes", "backslash", "tab", "hash-and-bracket", "micro", "u2028", "dot"],
)
def test_a_label_with_quotes_reads_back(original: str, label: str) -> None:
    # FM-190: a label is a turret label, whatever characters the stand's config used.
    section = ObjectivesSection(
        reference=label,
        offsets={
            f"other {label}": ObjectiveOffsetEntry(
                xy_offset_um=(1.5, -2.5), measured=MEASURED
            )
        },
    )

    written = with_objectives(original, section, where=WHERE)

    assert _objectives_of(written) == section
    # A second write finds the block it wrote, whatever the labels held.
    again = with_objectives(written, OTHER, where=WHERE)
    assert _objectives_of(again) == OTHER


def test_the_written_text_reads_back_as_the_section(original: str) -> None:
    sections = [
        SECTION,
        OTHER,
        ObjectivesSection(),
        ObjectivesSection(reference="only a reference"),
        ObjectivesSection(
            reference="R",
            offsets={
                "z": ObjectiveOffsetEntry(z_offset_um=1.25, measured=MEASURED),
                "both": ObjectiveOffsetEntry(
                    xy_offset_um=(0.0, -0.01), z_offset_um=-0.5, measured=MEASURED
                ),
                "nothing measured": ObjectiveOffsetEntry(),
            },
        ),
    ]

    for section in sections:
        assert (
            _objectives_of(with_objectives(original, section, where=WHERE)) == section
        )


def test_a_block_that_would_read_back_as_other_offsets_is_refused_and_nothing_is_returned(
    original: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = profile_write.objectives_block

    def another(section: ObjectivesSection, *, newline: str = "\n") -> str:
        return real(OTHER, newline=newline)

    monkeypatch.setattr(profile_write, "objectives_block", another)

    with pytest.raises(ProfileError) as raised:
        with_objectives(original, SECTION, where=WHERE)

    assert str(raised.value) == (
        f"{WHERE}: the text written for [objectives] does not read back as the "
        "offsets; nothing was written"
    )


def test_a_block_that_is_not_toml_is_refused_the_same_way(
    original: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        profile_write,
        "objectives_block",
        lambda section, *, newline="\n": f"{BLOCK_BEGIN}\nnot toml\n{BLOCK_END}\n",
    )

    with pytest.raises(ProfileError, match="does not read back as the offsets"):
        with_objectives(original, SECTION, where=WHERE)
