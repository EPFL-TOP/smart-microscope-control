"""The lab's own plate definitions: the data file, its loader and its registration with ``useq`` (m3 §22, #400).

A plate tool names its plate by a ``useq`` key, and a run document stores that
name, not the geometry. So the tests pin three things: every number of the
shipped file says where it comes from (D2), a published name keeps its numbers
(D5), and the names are seen from every entry point's fresh import (D3) without
ever redefining one of ``useq``'s own (D4).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator

import pytest
import useq
from useq import _plate_registry

from smc import plates
from smc.hardware.profile import SampleSection
from smc.plates import PLATES_FILE, LabPlate, PlateSource, load_lab_plates

#: Every published plate's numbers: ``(rows, columns, well_spacing_mm,
#: well_size_mm, circular_wells)``. A run document stores the plate's name, so
#: editing a published plate's numbers would move every well of every run that
#: names it, silently (D5). A corrected definition takes a new name, added here;
#: changing a line of this literal is a decision, made in the same commit.
PUBLISHED = {
    "greiner-24-well": (4, 6, (19.45, 19.45), (15.6, 15.6), True),
}

#: A valid plate, with ``{name}`` and ``{extra}`` to fill and lines to replace.
TEMPLATE = """\
[plates.{name}]
description = "a test plate"
rows = 4
columns = 6
well_spacing_mm = [19.45, 19.45]
well_size_mm = [15.6, 15.6]
circular_wells = true
well_spacing_source = {{ kind = "measured", note = "a bench, a ruler" }}
well_size_source = {{ kind = "assumed", note = "useq's 24-well" }}
{extra}"""


def _text(name: str = "test-24-well", extra: str = "", **replace: str) -> str:
    text = TEMPLATE.format(name=name, extra=extra)
    for old, new in replace.items():
        line = next(ln for ln in text.splitlines() if ln.startswith(f"{old} ="))
        text = text.replace(line, new)
    return text


@pytest.fixture
def registry() -> Iterator[dict[str, object]]:
    """``useq``'s registry, restored in place after the test (other modules hold the same dict)."""
    reg = _plate_registry._PLATE_REGISTRY
    saved = dict(reg)
    yield reg
    reg.clear()
    reg.update(saved)


def _plate(**changes: object) -> LabPlate:
    fields: dict[str, object] = {
        "description": "a test plate",
        "rows": 4,
        "columns": 6,
        "well_spacing_mm": (19.0, 19.0),
        "well_size_mm": (15.6, 15.6),
        "circular_wells": True,
        "well_spacing_source": PlateSource(kind="assumed", note="a test"),
        "well_size_source": PlateSource(kind="assumed", note="a test"),
    }
    fields.update(changes)
    return LabPlate.model_validate(fields)


# ---- the shipped file ------------------------------------------------------


def test_the_shipped_file_loads_and_every_plate_says_where_its_numbers_come_from() -> (
    None
):
    loaded = load_lab_plates()

    assert "greiner-24-well" in loaded
    for name, plate in loaded.items():
        for source in (plate.well_spacing_source, plate.well_size_source):
            assert source.kind in ("measured", "maker", "assumed"), name
            assert source.note.strip(), name


def test_a_published_plate_keeps_its_numbers() -> None:
    """D5: a new definition is a new name, added to ``PUBLISHED``; a published one never changes."""
    loaded = {
        name: (
            p.rows,
            p.columns,
            p.well_spacing_mm,
            p.well_size_mm,
            p.circular_wells,
        )
        for name, p in load_lab_plates().items()
    }

    assert loaded == PUBLISHED


def test_the_greiner_plate_says_its_spacing_was_measured_and_its_size_assumed() -> None:
    greiner = load_lab_plates()["greiner-24-well"]

    assert greiner.well_spacing_source.kind == "measured"
    assert "session 3" in greiner.well_spacing_source.note
    assert greiner.well_size_source.kind == "assumed"


# ---- the registration --------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    ["smc", "smc.cli", "smc.plugins.plate_walls.plugin", "smc.ui.controller"],
)
def test_a_lab_plate_is_seen_by_useq_from_a_fresh_import(module: str) -> None:
    """D3: every entry point imports ``smc`` first, so none has to remember to register."""
    code = (
        f"import {module}\n"
        "import useq\n"
        "plate = useq.WellPlate.from_str('greiner-24-well')\n"
        "print(plate.well_spacing, 'greiner-24-well' in useq.registered_well_plate_keys())\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert "(19.45, 19.45) True" in done.stdout, done.stderr


def test_a_bad_plates_file_fails_the_import_of_smc_naming_the_file_and_the_plate() -> (
    None
):
    """A plate whose numbers are wrong must not be used silently: the import itself fails.

    The packaged file is replaced, in a fresh interpreter, before ``smc`` is
    imported; a registration whose error was swallowed would print ``imported``.
    """
    bad = _text(name="greiner-24-well", rows="rows = 0")
    code = (
        "import importlib.resources\n"
        "class _File:\n"
        "    def joinpath(self, name):\n"
        "        return self\n"
        "    def read_text(self, encoding=None):\n"
        f"        return {bad!r}\n"
        "importlib.resources.files = lambda package: _File()\n"
        "import smc\n"
        "print('imported')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )

    assert done.returncode != 0, done.stdout
    assert "imported" not in done.stdout
    assert "plates.toml: plate 'greiner-24-well': rows:" in done.stderr, done.stderr


def test_useq_gets_each_axis_and_the_well_shape_as_filed(
    monkeypatch: pytest.MonkeyPatch, registry: dict[str, object]
) -> None:
    """FM-73: a pitch or a well that differs by axis reaches ``useq`` unswapped, square wells square."""
    monkeypatch.setattr(
        plates,
        "load_lab_plates",
        lambda text=None: {
            "test-anisotropic-15-well": _plate(
                rows=3,
                columns=5,
                well_spacing_mm=(19.45, 19.0),
                well_size_mm=(15.6, 15.0),
                circular_wells=False,
            )
        },
    )
    monkeypatch.setattr(plates, "_registered", None)

    plates.register_lab_plates()
    plate = useq.WellPlate.from_str("test-anisotropic-15-well")

    assert (plate.rows, plate.columns) == (3, 5)
    assert plate.well_spacing == (19.45, 19.0)
    assert plate.well_size == (15.6, 15.0)
    assert plate.circular_wells is False


def test_registering_twice_is_a_no_op_that_returns_the_same_names(
    monkeypatch: pytest.MonkeyPatch, registry: dict[str, object]
) -> None:
    calls: list[str | None] = []

    def load(text: str | None = None) -> dict[str, LabPlate]:
        calls.append(text)
        return {"test-twice-well": _plate()}

    monkeypatch.setattr(plates, "load_lab_plates", load)
    monkeypatch.setattr(plates, "_registered", None)

    first = plates.register_lab_plates()
    second = plates.register_lab_plates()

    assert first == second == ("test-twice-well",)
    assert len(calls) == 1
    assert "test-twice-well" in useq.registered_well_plate_keys()


def test_the_registered_plate_is_useqs_with_the_name_as_its_key() -> None:
    """``useq`` gets the geometry only: the key stays the plate's name, not the description."""
    assert plates.register_lab_plates() == tuple(sorted(PUBLISHED))

    plate = useq.WellPlate.from_str("greiner-24-well")

    assert plate.name == "greiner-24-well"
    assert (plate.rows, plate.columns) == (4, 6)
    assert plate.well_spacing == (19.45, 19.45)
    assert plate.well_size == (15.6, 15.6)
    assert plate.circular_wells is True


def test_a_lab_plate_cannot_redefine_a_useq_plate(
    monkeypatch: pytest.MonkeyPatch, registry: dict[str, object]
) -> None:
    """D4: ``useq`` overwrites a key without a word; a run document naming ``24-well`` would move."""
    monkeypatch.setattr(
        plates,
        "load_lab_plates",
        lambda text=None: {
            "a-new-24-well": _plate(),
            "24-well": _plate(well_spacing_mm=(19.45, 19.45)),
        },
    )
    monkeypatch.setattr(plates, "_registered", None)

    with pytest.raises(ValueError, match=r"plates\.toml.*'24-well'"):
        plates.register_lab_plates()

    assert useq.WellPlate.from_str("24-well").well_spacing == (19.0, 19.0)
    # All or none: the plate that did not collide is not registered either.
    assert "a-new-24-well" not in useq.registered_well_plate_keys()


# ---- the validation ------------------------------------------------------------


def test_the_template_of_the_bad_definitions_is_itself_valid() -> None:
    """So each bad text below fails for its one change, not for the template."""
    assert set(load_lab_plates(_text())) == {"test-24-well"}


@pytest.mark.parametrize(
    ("text", "name", "reason"),
    [
        (_text(name="test-24"), "test-24", "must end in '-well'"),
        (_text(name="Test-24-well"), "Test-24-well", "lower case"),
        (_text(extra="pitch_mm = 19.0\n"), "test-24-well", "pitch_mm"),
        (
            _text(well_spacing_source=""),
            "test-24-well",
            "well_spacing_source",
        ),
        (
            _text(
                well_size_source='well_size_source = { kind = "guessed", note = "x" }'
            ),
            "test-24-well",
            "kind",
        ),
        (
            _text(well_spacing_mm="well_spacing_mm = [0.0, 19.45]"),
            "test-24-well",
            "well_spacing_mm",
        ),
        (
            _text(well_spacing_mm="well_spacing_mm = [nan, 19.45]"),
            "test-24-well",
            "well_spacing_mm",
        ),
        (
            _text(well_size_mm="well_size_mm = [20.0, 20.0]"),
            "test-24-well",
            "below the spacing",
        ),
        (_text(name="test-24-well-2"), "test-24-well-2", "must end in '-well'"),
        (
            _text(well_spacing_mm="well_spacing_mm = [inf, 19.45]"),
            "test-24-well",
            "finite",
        ),
        (_text(rows="rows = 0"), "test-24-well", "rows: Input should be greater"),
        (_text(rows="rows = 65"), "test-24-well", "rows: Input should be less"),
        (
            _text(columns="columns = 0"),
            "test-24-well",
            "columns: Input should be greater",
        ),
        (
            _text(columns="columns = 97"),
            "test-24-well",
            "columns: Input should be less",
        ),
        (
            _text(well_size_mm="well_size_mm = [19.45, 19.45]"),
            "test-24-well",
            "below the spacing",
        ),
        (
            _text(
                well_spacing_source='well_spacing_source = { kind = "measured", note = "" }'
            ),
            "test-24-well",
            "well_spacing_source.note",
        ),
        (_text(rows='rows = "4"'), "test-24-well", "rows: Input should be a valid"),
        (
            _text(circular_wells="circular_wells = 1"),
            "test-24-well",
            "circular_wells: Input should be a valid",
        ),
    ],
    ids=[
        "no-well-suffix",
        "upper-case",
        "unknown-key",
        "missing-source",
        "guessed-source",
        "zero-spacing",
        "nan-spacing",
        "well-wider-than-pitch",
        "text-after-well",
        "inf-spacing",
        "no-rows",
        "too-many-rows",
        "no-columns",
        "too-many-columns",
        "well-as-wide-as-pitch",
        "empty-note",
        "rows-as-text",
        "circular-as-number",
    ],
)
def test_a_bad_definition_names_the_file_and_the_plate(
    text: str, name: str, reason: str
) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"{PLATES_FILE}: plate {name!r}: ")
    ) as caught:
        load_lab_plates(text)

    assert reason in str(caught.value)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("[plates\n", "not TOML"),
        ("[other]\n", "'other'"),
        ("plates = 3\n", "[plates]"),
    ],
    ids=["not-toml", "unknown-table", "plates-not-a-table"],
)
def test_a_file_that_is_not_a_plate_table_names_the_file(
    text: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(PLATES_FILE)) as caught:
        load_lab_plates(text)

    assert reason in str(caught.value)


# ---- the profile -----------------------------------------------------------------


def test_the_demo_sample_takes_a_lab_plate() -> None:
    """The profile's ``[sample] plate`` checks the ``-well`` suffix and ``useq``'s registry."""
    assert SampleSection(plate="greiner-24-well").plate == "greiner-24-well"
