"""The plate page's *Preset* region, without a browser (m5 §10.6, #319).

The widgets are driven the way a session drives them (FM-161): a click is
``button.clicks += 1``, which reaches a handler through the same ``_handler`` a
session's event does, and a field, the *Wells* box or the plate select is changed
with ``value=``, which is how a session's change arrives, never by calling the
handler. What this cannot show is a browser: the look of the region and of its note
after a load is for ``/ui-check`` (FM-153). Panel's warnings are errors here (FM-147),
and a handler that failed fails the test that did not expect it.

Every test works in ``tmp_path`` as the working directory with ``SMC_PROFILES`` unset,
so that a preset a page saves lands in ``tmp_path/profiles/presets`` and none is ever
written into the repository's ``profiles/presets``.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, Literal, cast

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult, WallTouch, WellRef
from smc.run import StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import PlatePage
from smc.ui.plate_page import PlatePageView

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

pytestmark = [
    pytest.mark.filterwarnings(
        "error::DeprecationWarning", "error::PendingDeprecationWarning"
    ),
    pytest.mark.usefixtures("handler_failures"),
]

STAND = "demo-sample"


# --- stub plugins: the workflow's names and slots, with the parameters that matter ---


class NoParams(PluginParams):
    pass


class NoResult(PluginResult):
    pass


class RegistrationParams(PluginParams):
    plate: str = "96-well"
    refs: tuple[WellRef, ...] = ()
    walls: tuple[WallTouch, ...] = ()
    # The page fills it from its *Plate lies* choice (m5 §16 D5), as for the real one.
    orientation: str = "not-turned"


class ScanPlanParams(PluginParams):
    wells: str
    coverage: float = 1.0
    overlap: float = 0.1


class CellsParams(PluginParams):
    detector: Literal["model", "blobs"] = "model"
    score_threshold: float = 0.5


class Tool:
    name = provides = "tool"
    summary = "A test tool."
    Params: type[PluginParams] = NoParams
    Result: type[PluginResult] = NoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PluginResult:
        return self.Result()


class Registration(Tool):
    name = "plate-registration"
    provides = "plate"
    Params = RegistrationParams
    Result = PlateResult
    requires = (XYStage,)

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PlateResult:
        return PlateResult(
            plate=params.plate,
            a1_center_xy_um=(0.0, 0.0),
            rotation_deg=0.0,
            method="seed",
            residual_um=None,
            scale_error=None,
            n_refs=len(params.refs),
        )


class ScanPlan(Tool):
    name = provides = "scan-plan"
    Params = ScanPlanParams
    reads = ("plate",)


class Cells(Tool):
    name = provides = "cells"
    Params = CellsParams


class Gate(Tool):
    """A tool that runs until the test lets it go: an activity that is going, for as long as needed."""

    name = provides = "gate"
    entered = threading.Event()
    release = threading.Event()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> PluginResult:
        Gate.entered.set()
        # Bounded (FM-44): a test that never lets it go ends the run on its own.
        Gate.release.wait(10.0)
        return self.Result()


WORKFLOW = (Registration, ScanPlan, Cells)

SUGGEST = {
    "scan-plan": {
        "params": ["coverage=0.4", "overlap=0.2"],
        "why": "fast on the demo stage",
    }
}

WEEKLY = """\
stand = "demo-sample"
plate = "96-well"
wells = "B2,B11"

[params]
scan-plan = ["coverage=0.7"]
cells = ["detector=blobs"]
"""

#: A preset that holds fields only: loading it changes neither the plate nor the wells.
FIELDS_ONLY = """\
[params]
scan-plan = ["coverage=0.7"]
"""


def _ep(cls: type[Tool]) -> EntryPoint:
    return EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMC_PROFILES", raising=False)
    Gate.entered.clear()
    Gate.release.clear()
    yield
    # A test that failed with the gate shut must not leave its run waiting.
    Gate.release.set()


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _profile(suggest: dict[str, Any] | None = SUGGEST) -> Profile:
    data: dict[str, Any] = {"microscope": {"name": STAND}}
    if suggest is not None:
        data["suggest"] = suggest
    return Profile.model_validate(data)


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    """A closed page over a fake stand; every page and stand is closed afterwards."""
    pages: list[PlatePage] = []
    stands: list[Microscope] = []

    def open_stand() -> Microscope:
        stand = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()),
            Profile.model_validate({"microscope": {"name": STAND}}),
        )
        stands.append(stand)
        return stand

    def make(
        profile: Profile | None = None,
        *,
        tools: tuple[type[Tool], ...] = WORKFLOW,
        connect: bool = False,
    ) -> PlatePage:
        page = PlatePage.closed(
            profile or _profile(),
            run_doc,
            opener=open_stand,
            entry_points=[_ep(cls) for cls in tools],
        )
        pages.append(page)
        if connect:
            page.connect()
            assert page.join(5), "connect did not end"
            assert page.connected, page.status().outcome
        return page

    try:
        yield make
    finally:
        for page in pages:
            page.close()
        for stand in stands:
            stand.close()


@contextmanager
def _replaced(obj: object, name: str, replacement: object) -> Iterator[None]:
    """An instance attribute for the length of the block, then removed (never kept: FM-92)."""
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        delattr(obj, name)


def _click(button: Any) -> None:
    button.clicks += 1


def _put(tmp_path: Path, name: str = "weekly", text: str = WEEKLY) -> Path:
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _load(view: PlatePageView, name: str = "weekly") -> None:
    """Choose ``name`` in the select and press *Load*, as a session would."""
    view.preset_select.value = name
    _click(view.preset_load_button)


def _plate_done(run_doc: Path) -> None:
    """The ``plate`` slot filled, so that the next step is the scan plan's."""
    then = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
    result = PlateResult(
        plate="96-well",
        a1_center_xy_um=(0.0, 0.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=1,
    ).model_dump(mode="json")
    write_stage(
        run_doc,
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a0",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile=STAND,
            dry_run=False,
            smc_version="0",
            params={"plate": "96-well"},
            result=result,
        ),
    )


def _command_words(view: PlatePageView) -> list[str]:
    """The words of the next step's ``command:`` line, as a POSIX shell splits them."""
    prefix = "  command: "
    (line,) = [
        line for line in view.next_text.object.split("\n") if line.startswith(prefix)
    ]
    return shlex.split(line.removeprefix(prefix))


def _field(view: PlatePageView, tool: str, key: str) -> Any:
    return view.tool_fields[tool][key]


# --- the region ---------------------------------------------------------------------


def test_the_preset_region_heads_the_expert_view_and_the_stand_screen(
    make_page: Callable[..., PlatePage],
) -> None:
    view = PlatePageView(make_page())

    # The first thing the expert view shows, and on the Stand screen after its title,
    # its lead and the next step: the same column, one instance (m5 §17.2 D5).
    expert = view.screens["expert"]
    region = expert[0]
    assert region in view.screens["stand"]
    assert region[0].object == "Preset"
    assert list(region[1]) == [view.preset_select, view.preset_load_button]
    assert list(region[2]) == [view.preset_name, view.preset_save_button]
    assert region[3] is view.preset_confirm
    assert region[4] is view.preset_note
    # Above the next step, and the first thing the column shows.
    assert expert[1][0].object == "Next step"
    # Spelled `label=` (FM-147) and every one there from the start.
    assert view.preset_select.name == "Preset"
    assert view.preset_load_button.name == "Load"
    assert view.preset_name.name == "Save as"
    assert view.preset_name.placeholder == "weekly"
    assert view.preset_save_button.name == "Save"
    assert view.preset_overwrite_button.name == "Overwrite"
    assert view.preset_cancel_button.name == "Cancel"
    assert not view.preset_confirm.visible
    assert view.preset_note.object == "no preset loaded"
    assert view.preset_select.options == {"(none)": ""}
    # The note and the question hold paths with no space in them: both wrap anywhere,
    # which `pre-wrap` alone did not (the question was cut at the column's edge,
    # seen with /ui-check, FM-153).
    for text in (view.preset_note, view.preset_confirm_text):
        assert text.styles["overflow-wrap"] == "anywhere"
        assert text.styles["white-space"] == "pre-wrap"


def test_the_options_are_the_presets_of_the_folders_and_leave_a_broken_file_out(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    _put(tmp_path, "beta", FIELDS_ONLY)
    _put(tmp_path, "alpha", FIELDS_ONLY)
    _put(tmp_path, "broken", "wells = \n")

    view = PlatePageView(make_page())

    assert view.preset_select.options == {
        "(none)": "",
        "alpha": "alpha",
        "beta": "beta",
    }
    assert view.preset_select.value == ""


# --- load and the note --------------------------------------------------------------


def test_load_fills_every_field_from_the_preset_then_the_suggestion_then_empty(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc)
    path = _put(tmp_path)
    page = make_page()
    view = PlatePageView(page)
    # The page starts with the profile's suggestion in its fields.
    assert _field(view, "scan-plan", "overlap").value == "0.2"
    # Typed before the load: a load is choosing the preset's values, so these go.
    _field(view, "cells", "score_threshold").value = "0.9"
    _field(view, "scan-plan", "coverage").value = "9"

    _load(view)

    assert _field(view, "scan-plan", "coverage").value == "0.7"  # the preset's
    assert _field(view, "scan-plan", "overlap").value == "0.2"  # the suggestion's
    assert _field(view, "cells", "detector").value == "blobs"  # the preset's
    assert _field(view, "cells", "score_threshold").value == ""  # nobody's
    assert page.selection == ("B2", "B11")
    assert view.select_input.value == "B2,B11"
    assert view.preset_name.value == "weekly"
    assert view.message.object == ""
    assert view.preset_note.object == (
        f"loaded: weekly  ({path})\nthe page holds the preset's values"
    )
    # The next step reads the same values: the profile's key the preset does not give,
    # then the preset's.
    assert view.next_text.object is not None
    assert "overlap=0.2: suggested by the profile demo-sample" in view.next_text.object
    assert "coverage=0.7: from the preset weekly" in view.next_text.object
    # Read back the way a shell reads it: a Windows path has backslashes, which
    # `shlex.join` quotes, so the raw text of the path is not in the line there (FM-24).
    words = _command_words(view)
    assert words[words.index("--preset") + 1] == str(path)


def test_a_field_changed_through_a_session_is_listed_in_the_note(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(tmp_path)
    view = PlatePageView(make_page())
    _load(view)
    holds = f"loaded: weekly  ({path})\nthe page holds the preset's values"
    assert view.preset_note.object == holds

    # A change arrives as `value=`, which is what a session's does (FM-161).
    _field(view, "scan-plan", "coverage").value = " 0.5 "
    assert view.preset_note.object == (
        f"loaded: weekly  ({path})\nchanged since loading:\n"
        "  scan-plan coverage: 0.5 (the preset: 0.7)"
    )
    _field(view, "cells", "detector").value = ""
    assert view.preset_note.object.endswith(
        "  cells detector: (default) (the preset: blobs)"
    )
    # The edit undone: the note follows.
    _field(view, "scan-plan", "coverage").value = "0.7"
    _field(view, "cells", "detector").value = "blobs"
    assert view.preset_note.object == holds


def test_the_wells_box_and_the_plate_select_redraw_the_note_too(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(tmp_path)
    view = PlatePageView(make_page())
    _load(view)

    view.select_input.value = "A1"
    assert view.preset_note.object == (
        f"loaded: weekly  ({path})\nchanged since loading:\n"
        "  wells: 1 selected (the preset: B2,B11)"
    )

    # Another plate drops the selection: both lines say so.
    view.plate_select.value = "384-well"
    assert view.preset_note.object == (
        f"loaded: weekly  ({path})\nchanged since loading:\n"
        "  plate: 384-well (the preset: 96-well)\n"
        "  wells: 0 selected (the preset: B2,B11)"
    )


def test_the_note_is_drawn_once_for_a_load_of_many_fields(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    _put(tmp_path)
    view = PlatePageView(make_page())
    drawn: list[str] = []
    inside: list[int] = []
    real_note = view._draw_preset_note
    real_fill = view._fill_fields

    def note() -> None:
        drawn.append("drawn")
        real_note()

    def fill(loaded: Any) -> None:
        before = len(drawn)
        real_fill(loaded)
        inside.append(len(drawn) - before)

    with (
        _replaced(view, "_draw_preset_note", note),
        _replaced(view, "_fill_fields", fill),
    ):
        _load(view)

    # Two fields of the load change value (the preset's coverage and detector): while
    # they are assigned the watchers draw nothing, and the note is drawn after the
    # last, by the load's own `refresh()`.
    assert inside == [0]
    assert len(drawn) >= 1


def test_a_field_watcher_is_quiet_while_a_load_fills_the_fields_and_never_after(
    make_page: Callable[..., PlatePage], tmp_path: Path, handler_failures: list[Any]
) -> None:
    _put(tmp_path)
    view = PlatePageView(make_page())

    class Refusing:
        """A field whose assignment fails half-way through the load."""

        def __init__(self, real: Any) -> None:
            self.real = real

        @property
        def value(self) -> str:
            return str(self.real.value)

        @value.setter
        def value(self, new: str) -> None:
            raise RuntimeError("the browser went away")

    view.tool_fields["cells"]["detector"] = Refusing(
        view.tool_fields["cells"]["detector"]
    )
    _load(view)
    # The handler failed as handlers do: a line of the page, a record in the log.
    assert "that action failed on the page: RuntimeError" in view.message.object
    assert len(handler_failures) == 1
    handler_failures.clear()

    # The guard is down again: the note follows an edit once more.
    assert view._filling is False
    _field(view, "scan-plan", "coverage").value = "0.5"
    assert "scan-plan coverage: 0.5 (the preset: 0.7)" in view.preset_note.object


def test_load_none_forgets_the_preset(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    _put(tmp_path)
    page = make_page()
    view = PlatePageView(page)
    _load(view)
    assert page.preset is not None

    _load(view, "")

    assert page.preset is None
    assert view.preset_note.object == "no preset loaded"
    assert "--preset" not in view.next_text.object
    # The fields stay as they are: forgetting is not clearing.
    assert _field(view, "scan-plan", "coverage").value == "0.7"


def test_a_preset_the_captures_refuse_is_loaded_with_its_mismatch_in_the_note(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(tmp_path, text=WEEKLY.replace("96-well", "384-well"))
    page = make_page()
    page.refs = (WellRef(well="A1", x_um=1.0, y_um=2.0),)
    view = PlatePageView(page)

    _load(view)

    assert page.plate == "96-well"
    assert view.preset_note.object == (
        f"loaded: weekly  ({path})\n"
        "  preset weekly is for a 384-well plate; this page has captures on a "
        "96-well plate: its plate and wells are not applied\n"
        "changed since loading:\n"
        "  plate: 96-well (the preset: 384-well)"
    )


def test_a_preset_text_with_markup_is_shown_as_text(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    # FM-139: a name, a stand and a path come from a file a person wrote.
    _put(tmp_path, text='stand = "<img src=x onerror=alert(1)>"\n')
    view = PlatePageView(make_page())

    _load(view)

    assert isinstance(view.preset_note, pn.pane.Str)
    assert (
        "preset weekly was saved on <img src=x onerror=alert(1)>; this stand is "
        "demo-sample"
    ) in view.preset_note.object


# --- save and overwrite -------------------------------------------------------------


def _saved(tmp_path: Path, name: str) -> dict[str, Any]:
    path = tmp_path / "profiles" / "presets" / f"{name}.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_save_writes_the_name_the_wells_and_the_fields_and_selects_it(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    view.select_input.value = "A1:B2"
    _field(view, "scan-plan", "coverage").value = "0.3"
    view.preset_name.value = "mine"

    _click(view.preset_save_button)

    path = tmp_path / "profiles" / "presets" / "mine.toml"
    stored = _saved(tmp_path, "mine")
    assert stored["wells"] == "A1:B2"
    # The page's suggestion is a typed field like any other: it is saved with it.
    assert stored["params"]["scan-plan"] == ["overlap=0.2", "coverage=0.3"]
    assert view.message.object == f"saved mine to {path}"
    assert view.preset_select.options == {"(none)": "", "mine": "mine"}
    assert view.preset_select.value == "mine"
    assert view.preset_note.object == (
        f"loaded: mine  ({path})\nthe page holds the preset's values"
    )
    assert not view.preset_confirm.visible


def test_save_with_no_name_says_so_and_writes_nothing(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    view.preset_name.value = "   "

    _click(view.preset_save_button)

    assert view.message.object == "type a name to save the preset under"
    assert not (tmp_path / "profiles").exists()


def test_save_on_an_existing_name_arms_the_confirmation_and_overwrite_saves(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    _field(view, "scan-plan", "coverage").value = "0.3"
    view.preset_name.value = "mine"
    _click(view.preset_save_button)
    path = tmp_path / "profiles" / "presets" / "mine.toml"
    first = path.read_bytes()

    # Another value, the same name: asked, not done.
    _field(view, "scan-plan", "coverage").value = "0.6"
    _click(view.preset_save_button)

    assert view.preset_confirm.visible
    assert (
        view.preset_confirm_text.object == f"preset mine exists: {path}: overwrite it?"
    )
    assert path.read_bytes() == first
    assert view._preset_armed == ("mine",)

    # The field is changed again before the click: the click saves what is shown then.
    _field(view, "scan-plan", "coverage").value = "0.8"
    _click(view.preset_overwrite_button)

    assert _saved(tmp_path, "mine")["params"]["scan-plan"] == [
        "overlap=0.2",
        "coverage=0.8",
    ]
    assert not view.preset_confirm.visible
    assert view._preset_armed is None
    assert view.message.object == f"saved mine to {path}"


def test_cancel_overwrites_nothing(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    view.preset_name.value = "mine"
    _click(view.preset_save_button)
    path = tmp_path / "profiles" / "presets" / "mine.toml"
    before = path.read_bytes()
    _field(view, "scan-plan", "coverage").value = "0.6"
    _click(view.preset_save_button)
    assert view.preset_confirm.visible

    _click(view.preset_cancel_button)

    assert not view.preset_confirm.visible
    assert view._preset_armed is None
    assert path.read_bytes() == before


def test_overwrite_after_the_name_changed_saves_nothing(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    view.preset_name.value = "mine"
    _click(view.preset_save_button)
    path = tmp_path / "profiles" / "presets" / "mine.toml"
    before = path.read_bytes()
    _field(view, "scan-plan", "coverage").value = "0.6"
    _click(view.preset_save_button)
    assert view.preset_confirm.visible

    # Another name: the confirmation goes with the change, and a click on a button a
    # stale tab still shows saves nothing.
    view.preset_name.value = "other"
    assert not view.preset_confirm.visible
    assert view._preset_armed is None
    _click(view.preset_overwrite_button)

    assert path.read_bytes() == before
    assert not (tmp_path / "profiles" / "presets" / "other.toml").exists()
    assert view.message.object == "nothing to overwrite: press Save again"


def test_an_overwrite_armed_for_another_name_than_the_box_saves_nothing(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    view.preset_name.value = "mine"
    _click(view.preset_save_button)
    path = tmp_path / "profiles" / "presets" / "mine.toml"
    before = path.read_bytes()
    # What a name changed behind the watcher's back would look like.
    view._preset_armed = ("mine",)
    with _replaced(view, "preset_name", pn.widgets.TextInput(value="other")):
        _click(view.preset_overwrite_button)

    assert path.read_bytes() == before
    assert view.message.object == "the name changed: press Save again"


# --- bounds and errors --------------------------------------------------------------


def test_a_value_the_tool_refuses_is_the_page_message_and_nothing_is_written(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    view = PlatePageView(make_page())
    _field(view, "scan-plan", "coverage").value = "lots"
    view.preset_name.value = "mine"

    _click(view.preset_save_button)

    assert view.message.object.startswith("scan-plan: ")
    assert view.message.object.endswith("; nothing saved")
    assert not (tmp_path / "profiles").exists()
    assert not view.preset_confirm.visible


def test_a_preset_error_is_the_page_message(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    # Through the handler a session registered and `_try`, not a log line (FM-98,
    # FM-171): `handler_failures` fails this test if a handler raised.
    path = _put(tmp_path)
    view = PlatePageView(make_page())
    path.unlink()

    _load(view)
    assert view.message.object.startswith("no preset named weekly; looked in ")
    assert view.message.object.endswith("; smc presets list shows the presets")
    assert view.page.preset is None

    view.preset_name.value = "My Preset"
    _click(view.preset_save_button)
    assert view.message.object == (
        "a preset's name is 1 to 64 lower-case letters, digits, - and _, starting "
        "with a letter or a digit; got 'My Preset'"
    )


def test_the_preset_options_are_never_read_by_the_tick(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    # FM-151: the folders are listed by `refresh()` and after a save, never every second.
    _put(tmp_path)
    page = make_page()
    view = PlatePageView(page)
    calls: list[str] = []
    real = page.presets

    def spy() -> Any:
        calls.append("listed")
        return real()

    with _replaced(page, "presets", spy):
        for _ in range(5):
            view.tick()
        assert calls == []
        view.refresh()
        assert calls == ["listed"]
        view.preset_name.value = "mine"
        _click(view.preset_save_button)
        # The save lists them for the select and `refresh()` lists them once more.
        assert len(calls) <= 3


def test_no_preset_control_is_disabled_while_a_run_goes_and_each_one_works(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    # None of them reaches the stand (m5 §10.6). The tools' buttons are greyed while an
    # activity goes: that one is the contrast that shows the region is not.
    _put(tmp_path)
    page = make_page(tools=(Gate, *WORKFLOW), connect=True)
    view = PlatePageView(page)
    page.run("gate", [])
    assert Gate.entered.wait(5.0), "the run did not start"
    try:
        view.tick()
        assert view._running == "gate"
        assert view.tool_runs["cells"].disabled
        region = (
            view.preset_select,
            view.preset_load_button,
            view.preset_name,
            view.preset_save_button,
            view.preset_overwrite_button,
            view.preset_cancel_button,
        )
        assert not any(widget.disabled for widget in region)

        _load(view)
        assert page.preset is not None
        assert page.preset.file.name == "weekly"
        view.preset_name.value = "mine"
        _click(view.preset_save_button)
        assert view.message.object.startswith("saved mine to ")
        assert page.preset.file.name == "mine"
        # Still going: nothing the region did ended or refused the run.
        assert page.status().running == "gate"
    finally:
        Gate.release.set()
    assert page.join(10.0), "the run did not end"


# --- another tab --------------------------------------------------------------------


def test_a_second_tab_names_the_preset_the_first_loaded_and_how_its_fields_differ(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc)
    path = _put(tmp_path, text=FIELDS_ONLY)
    page = make_page()
    first = PlatePageView(page)
    second = PlatePageView(page)
    assert second.preset_note.object == "no preset loaded"
    # The second tab's own field: it keeps its own, as every tab does.
    _field(second, "scan-plan", "coverage").value = "0.5"

    _load(first)
    # The load changed neither the plate nor the wells: only the preset's identity
    # tells the second tab (FM-151). Its tick redraws.
    second.tick()

    assert second.preset_note.object == (
        f"loaded: weekly  ({path})\nchanged since loading:\n"
        "  scan-plan coverage: 0.5 (the preset: 0.7)"
    )
    assert "coverage=0.7: from the preset weekly" in second.next_text.object
    assert first.preset_note.object == (
        f"loaded: weekly  ({path})\nthe page holds the preset's values"
    )


def test_a_second_tab_lists_a_preset_the_first_saved(
    make_page: Callable[..., PlatePage],
) -> None:
    page = make_page()
    first = PlatePageView(page)
    second = PlatePageView(page)
    first.preset_name.value = "mine"

    _click(first.preset_save_button)
    second.tick()

    assert second.preset_select.options == {"(none)": "", "mine": "mine"}


def test_a_handler_failure_log_is_a_line_of_the_page(
    make_page: Callable[..., PlatePage], handler_failures: list[logging.LogRecord]
) -> None:
    # The fixture's own contract, here for the region: a button whose controller raises
    # something that is not an operator's mistake becomes the line, not a console.
    page = make_page()
    view = PlatePageView(page)

    def boom(source: str) -> Any:
        raise KeyError("boom")

    view.preset_select.options = {"(none)": "", "x": "x"}
    with _replaced(page, "load_preset", boom):
        _load(view, "x")

    assert view.message.object.startswith("that action failed on the page: KeyError")
    assert len(handler_failures) == 1
    handler_failures.clear()
