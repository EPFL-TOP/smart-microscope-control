"""Which way the plate lies, on the page: the select beside the plate menu, the stand's sides and A1 (m5 §16, #422).

Panel widgets and Bokeh sources only, no browser: the page is built over ``FakeCore`` on
the demo profile with a ``[holder]`` whose sides are set or not. The expected places are
written from the 96-well's numbers (9 mm pitch, 6.4 mm wells, A1 at the holder's centre
minus 5.5 by 3.5 pitches), never read back from the code under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import panel as pn
import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.profile import HolderSection, Profile
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.guide import ORIENTATION_IN_STAGE_TERMS
from smc.ui.plate_page import PlatePageView

#: A failure a handler swallowed fails the test that did not expect it (#277).
pytestmark = pytest.mark.usefixtures("handler_failures")

DEG = "\N{DEGREE SIGN}"
HOLDER_CENTRE = (400.0, -325.0)
#: The 96-well estimate's wells' box: A1 at (-49 100, 31 175) µm, 11 and 7 pitches of
#: 9 000 µm to the last column and row, and a 3 200 µm radius around each centre.
LEFT, RIGHT = -49_100.0 - 3_200.0, -49_100.0 + 99_000.0 + 3_200.0
TOP, BOTTOM = 31_175.0 + 3_200.0, 31_175.0 - 63_000.0 - 3_200.0
HALF_PITCH = 4_500.0


def _profile(sides: tuple[str, str] | None) -> Profile:
    extra: dict[str, Any] = (
        {}
        if sides is None
        else {
            "back": sides[0],
            "right": sides[1],
            "directions_measured": "2026-11-11, a test",
        }
    )
    holder = HolderSection(
        plate_centre_xy_um=HOLDER_CENTRE, measured="2026-11-10, a test", **extra
    )
    return Profile.demo().model_copy(update={"holder": holder})


@pytest.fixture
def make_view(fake_core: FakeCore, tmp_path: Path) -> Iterator[Any]:
    made: list[tuple[PlatePage, Microscope]] = []

    def make(sides: tuple[str, str] | None = ("+Y", "+X")) -> PlatePageView:
        microscope = Microscope.from_core(
            cast("CMMCorePlus", fake_core), _profile(sides)
        )
        page = PlatePage(microscope, tmp_path / "run.json", entry_points=[])
        made.append((page, microscope))
        page.choose_plate("96-well")
        view = PlatePageView(page)
        view.refresh()
        return view

    try:
        yield make
    finally:
        for page, microscope in made:
            page.close()
            microscope.close()


def _labels(source: Any) -> dict[str, tuple[float, float]]:
    data = source.data
    return {
        text: (float(x), float(y))
        for text, x, y in zip(data["text"], data["x"], data["y"], strict=True)
    }


def _close(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6


def test_the_orientation_select_is_next_to_the_plate_menu(make_view: Any) -> None:
    view = make_view()
    rows = [row for row in view.layout.select(pn.Row) if view.plate_select in row]
    assert len(rows) == 1
    assert list(rows[0])[:2] == [view.plate_select, view.orientation_select]
    select = view.orientation_select
    assert select.name == "Plate lies"
    # The Ti2-E's words: +Y is the back and +X the operator's right.
    assert select.options == {
        "A1 at the back left": "not-turned",
        f"Turned 180{DEG}: A1 at the front right": "turned-180",
    }
    assert select.value == "not-turned"
    assert not view.orientation_note.visible


def test_the_map_labels_the_stands_sides_and_a1(make_view: Any) -> None:
    view = make_view()
    middle_x, middle_y = (LEFT + RIGHT) / 2.0, (TOP + BOTTOM) / 2.0
    sides = _labels(view.side_source)
    assert set(sides) == {
        "back of the stand",
        "front of the stand",
        "operator's left",
        "operator's right",
    }
    assert _close(sides["back of the stand"], (middle_x, TOP + HALF_PITCH))
    assert _close(sides["front of the stand"], (middle_x, BOTTOM - HALF_PITCH))
    assert _close(sides["operator's right"], (RIGHT + HALF_PITCH, middle_y))
    assert _close(sides["operator's left"], (LEFT - HALF_PITCH, middle_y))
    assert _labels(view.a1_source) == {"A1": (-49_100.0, 31_175.0)}

    # Turned, the sides stay where the stand's are and A1 moves to the far corner.
    view.orientation_select.value = "turned-180"
    turned = _labels(view.side_source)
    assert set(turned) == set(sides)
    assert all(_close(turned[text], sides[text]) for text in sides)
    assert _close(
        _labels(view.a1_source)["A1"], (-49_100.0 + 99_000.0, 31_175.0 - 63_000.0)
    )
    assert view.figure.title.text.endswith(
        f"declared turned 180{DEG}, A1 at the bottom right of this map"
    )


def test_a_stand_whose_back_is_minus_x_has_it_on_the_left_of_the_map(
    make_view: Any,
) -> None:
    # A word found by its axis, or by the sign of Y, would put the back at the top
    # (FM-73): here the back is -X and the operator's right +Y.
    view = make_view(("-X", "+Y"))
    middle_x, middle_y = (LEFT + RIGHT) / 2.0, (TOP + BOTTOM) / 2.0
    sides = _labels(view.side_source)
    assert _close(sides["back of the stand"], (LEFT - HALF_PITCH, middle_y))
    assert _close(sides["front of the stand"], (RIGHT + HALF_PITCH, middle_y))
    assert _close(sides["operator's right"], (middle_x, TOP + HALF_PITCH))
    assert _close(sides["operator's left"], (middle_x, BOTTOM - HALF_PITCH))
    # A1 at -X +Y is then the back right.
    assert next(iter(view.orientation_select.options)) == "A1 at the back right"


@pytest.mark.parametrize("plate", ["384-well", "1536-well"])
def test_a1_is_labelled_on_a_384_well_plate(make_view: Any, plate: str) -> None:
    view = make_view()
    view.plate_select.value = plate
    assert view.page.plate == plate
    # The other names are noise past 96 wells; A1 is the one the operator checks.
    assert not view.names.visible
    a1 = _labels(view.a1_source)
    assert list(a1) == ["A1"]
    centres = {
        name: (x, y)
        for name, x, y in zip(
            view.wells_source.data["name"],
            view.wells_source.data["x"],
            view.wells_source.data["y"],
            strict=True,
        )
    }
    assert _close(a1["A1"], centres["A1"])


def test_without_directions_the_map_has_no_side_labels_and_says_why(
    make_view: Any,
) -> None:
    view = make_view(None)
    assert _labels(view.side_source) == {}
    assert view.orientation_note.visible
    assert view.orientation_note.object == ORIENTATION_IN_STAGE_TERMS
    assert view.orientation_select.options == {
        "Not turned: A1 at stage -X +Y": "not-turned",
        f"Turned 180{DEG}: A1 at stage +X -Y": "turned-180",
    }
    # A1 is labelled all the same.
    assert list(_labels(view.a1_source)) == ["A1"]


def test_the_select_is_disabled_while_an_activity_goes(
    make_view: Any,
) -> None:
    view = make_view()
    view._running = "scan"
    view._draw_buttons()
    assert view.orientation_select.disabled
    view._running = None
    view._draw_buttons()
    assert not view.orientation_select.disabled
