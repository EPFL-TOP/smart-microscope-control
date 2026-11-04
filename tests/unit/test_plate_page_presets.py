"""The plate page's presets, in the controller: load, save, the record and the next step (m5 §10.5, #319).

No Panel here. The stand is a ``FakeCore`` handed to the page by an opener and the
tools are stubs with the workflow's names and slots, as in ``test_plate_page_next_step``.
Every test works in ``tmp_path`` as the working directory with ``SMC_PROFILES`` unset,
so a preset saved by a page lands in ``tmp_path/profiles/presets`` and none is ever
written into the repository's ``profiles/presets``. Texts are asserted whole (FM-45).
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable, Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.profile import Profile
from smc.plugins.api import PluginParams, PluginResult, RunContext
from smc.plugins.presets import (
    NOT_IN_PARAMS,
    PresetError,
    PresetExistsError,
    load_preset,
)
from smc.plugins.registry import GROUP
from smc.plugins.slots import PlateResult, WallTouch, WellRef
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore
from smc.ui.controller import PAGE_PARAMS, PlatePage

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

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
    summary = "A test registration."
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


WORKFLOW = (Registration, ScanPlan, Cells)


def _ep(cls: type[Tool]) -> EntryPoint:
    return EntryPoint(name=cls.name, value=f"{__name__}:{cls.__name__}", group=GROUP)


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The preset folders are the test's own: ``./profiles/presets`` and ``./presets``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMC_PROFILES", raising=False)


@pytest.fixture
def run_doc(tmp_path: Path) -> Path:
    return tmp_path / "run.json"


def _profile(suggest: dict[str, Any] | None = None) -> Profile:
    data: dict[str, Any] = {"microscope": {"name": STAND}}
    if suggest is not None:
        data["suggest"] = suggest
    return Profile.model_validate(data)


@pytest.fixture
def make_page(run_doc: Path) -> Iterator[Callable[..., PlatePage]]:
    """A page over a fake stand, closed afterwards with its stand."""
    pages: list[PlatePage] = []
    stands: list[Microscope] = []

    def open_stand() -> Microscope:
        stand = Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()),
            Profile.model_validate({"microscope": {"name": STAND}}),
        )
        stands.append(stand)
        return stand

    def make(profile: Profile | None = None, *, connect: bool = False) -> PlatePage:
        page = PlatePage.closed(
            profile or _profile(),
            run_doc,
            opener=open_stand,
            entry_points=[_ep(cls) for cls in WORKFLOW],
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


WEEKLY = """\
stand = "demo-sample"
plate = "96-well"
wells = "B2,B11"

[params]
scan-plan = ["coverage=0.4"]
cells = ["detector=blobs"]
"""


def _put(tmp_path: Path, name: str = "weekly", text: str = WEEKLY) -> Path:
    folder = tmp_path / "profiles" / "presets"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _plate_done(run_doc: Path, plate: str = "96-well") -> None:
    from datetime import datetime, timezone

    then = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
    result = PlateResult(
        plate=plate,
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
            params={"plate": plate},
            result=result,
        ),
    )


# --- the names the page fills and the names a preset does not keep ------------------


def test_what_a_preset_does_not_keep_under_params_is_what_the_page_fills() -> None:
    # Two lists that must agree: a key the page fills itself is one the preset's
    # own fields hold, and the other way round.
    assert {plugin: set(keys) for plugin, keys in NOT_IN_PARAMS.items()} == {
        plugin: set(keys) for plugin, keys in PAGE_PARAMS.items()
    }


# --- loading ----------------------------------------------------------------------


def test_load_preset_applies_the_plate_the_wells_and_the_fields(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(tmp_path, text=WEEKLY.replace("96-well", "384-well"))
    page = make_page()
    assert page.plate == "96-well"

    loaded = page.load_preset(str(path))

    assert page.plate == "384-well"
    assert page.selection == ("B2", "B11")
    assert loaded.selection == ("B2", "B11")
    assert loaded.fields == {
        "scan-plan": {"coverage": "0.4"},
        "cells": {"detector": "blobs"},
    }
    assert loaded.mismatches == ()
    assert (loaded.file.name, loaded.file.path) == ("weekly", path)
    assert page.preset is loaded


def test_load_preset_by_name_looks_in_the_stands_folders(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    _put(tmp_path)
    page = make_page()

    assert [entry.name for entry in page.presets()] == ["weekly"]
    assert page.load_preset("weekly").file.name == "weekly"


def test_load_preset_keeps_the_captures_and_names_the_plate_mismatch(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(tmp_path, text=WEEKLY.replace("96-well", "384-well"))
    page = make_page()
    page.select_text("A1")
    page.refs = (WellRef(well="A1", x_um=1.0, y_um=2.0),)

    loaded = page.load_preset(str(path))

    # `choose_plate` would have dropped the captures: nothing about the plate moves.
    assert page.plate == "96-well"
    assert page.refs == (WellRef(well="A1", x_um=1.0, y_um=2.0),)
    assert page.selection == ("A1",)
    assert loaded.selection is None
    assert loaded.mismatches == (
        "preset weekly is for a 384-well plate; this page has captures on a "
        "96-well plate: its plate and wells are not applied",
    )
    # The fields are applied all the same: they hold no position.
    assert loaded.fields["scan-plan"] == {"coverage": "0.4"}


def test_load_preset_names_wells_that_do_not_parse_and_keeps_the_selection(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(tmp_path, text=WEEKLY.replace("B2,B11", "Z99"))
    page = make_page()
    page.select_text("A1")

    loaded = page.load_preset(str(path))

    assert page.selection == ("A1",)
    assert loaded.selection is None
    assert loaded.mismatches == (
        "preset weekly: wells Z99: wells not understood on a 96-well plate: 'Z99'",
    )


def test_load_preset_lists_what_the_check_found_then_the_pages_own(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    path = _put(
        tmp_path,
        text=(
            WEEKLY.replace("demo-sample", "nikon-ti2")
            .replace("B2,B11", "Z99")
            .replace('"coverage=0.4"', '"coverage=0.4", "bogus=1"')
        ),
    )

    loaded = make_page().load_preset(str(path))

    assert loaded.mismatches == (
        "preset weekly was saved on nikon-ti2; this stand is demo-sample",
        "preset weekly [params.scan-plan]: bogus is not a parameter of scan-plan "
        "(keys: wells, coverage, overlap)",
        "preset weekly: wells Z99: wells not understood on a 96-well plate: 'Z99'",
    )
    assert loaded.fields["scan-plan"] == {"coverage": "0.4"}


def test_a_plate_the_run_document_registered_otherwise_is_not_applied(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc, "384-well")
    path = _put(tmp_path)  # a 96-well preset
    page = make_page()
    assert page.plate == "384-well"

    loaded = page.load_preset(str(path))

    assert page.plate == "384-well"
    assert page.selection == ()
    assert loaded.mismatches == (
        "preset weekly is for a 96-well plate; this run document's plate is "
        "registered as a 384-well: its plate and wells are not applied",
    )


def test_an_unknown_preset_raises_and_keeps_the_loaded_one(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()
    loaded = page.load_preset(str(_put(tmp_path)))

    with pytest.raises(PresetError, match=r"^no preset named nope; looked in "):
        page.load_preset("nope")

    assert page.preset is loaded


def test_a_load_is_allowed_while_the_stand_is_not_connected(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page(connect=False)
    assert not page.connected

    assert page.load_preset(str(_put(tmp_path))).file.name == "weekly"


# --- the record -------------------------------------------------------------------


def test_a_run_records_the_loaded_preset_name(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    page = make_page(connect=True)
    page.load_preset(str(_put(tmp_path)))

    page.run("plate-registration", ["plate=96-well"])
    assert page.join(10.0)

    record = RunDocument.load(run_doc).stages["plate-registration"]
    assert record.status is StageStatus.done
    assert record.preset == "weekly"


def test_after_forget_preset_a_run_records_none(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    page = make_page(connect=True)
    page.load_preset(str(_put(tmp_path)))
    page.forget_preset()
    assert page.preset is None

    page.run("plate-registration", ["plate=96-well"])
    assert page.join(10.0)

    assert RunDocument.load(run_doc).stages["plate-registration"].preset is None


# --- saving -----------------------------------------------------------------------


def _saved(tmp_path: Path, name: str) -> dict[str, Any]:
    """The file a save wrote, read back as TOML: the layout is tomli-w's, the content is ours."""
    path = tmp_path / "profiles" / "presets" / f"{name}.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_save_preset_keeps_the_typed_wells_when_they_name_the_selection(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()
    page.select_text("A1:B2")

    saved = page.save_preset("mine", {"scan-plan": ["coverage=0.4"]}, " A1:B2 ")

    stored = _saved(tmp_path, "mine")
    assert {key: stored[key] for key in ("stand", "plate", "wells", "params")} == {
        "stand": STAND,
        "plate": "96-well",
        "wells": "A1:B2",
        "params": {"scan-plan": ["coverage=0.4"]},
    }
    assert stored["saved_utc"].microsecond == 0
    assert stored["saved_utc"].utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    assert saved.selection == page.selection


def test_save_preset_joins_the_selection_when_the_box_says_otherwise(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()
    page.select_text("A1,A2")

    page.save_preset("mine", {}, "C*")
    page.save_preset("again", {}, "A1,Z99")  # a box that does not even parse

    assert _saved(tmp_path, "mine")["wells"] == "A1,A2"
    assert _saved(tmp_path, "again")["wells"] == "A1,A2"


def test_save_preset_with_no_selection_keeps_no_wells(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()

    # `scan-plan` needs a well to be looked at alone: the page adds a stand-in for the
    # check and the preset keeps none.
    page.save_preset("mine", {"scan-plan": ["coverage=0.4"]}, "A1")

    stored = _saved(tmp_path, "mine")
    assert "wells" not in stored
    assert stored["params"] == {"scan-plan": ["coverage=0.4"]}
    loaded = load_preset("mine", page._profile).preset
    assert loaded is not None
    assert loaded.wells is None


def test_save_preset_refuses_a_value_the_plugin_refuses_and_writes_nothing(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()
    page.select_text("A1")

    with pytest.raises(PresetError) as refused:
        page.save_preset("mine", {"scan-plan": ["coverage=lots"]}, "A1")

    message = str(refused.value)
    assert message.startswith("scan-plan: ")
    assert "coverage" in message
    assert message.endswith("; nothing saved")
    assert not (tmp_path / "profiles").exists()
    assert page.preset is None


def test_save_preset_refuses_an_unknown_plugin_and_a_key_the_page_fills(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()

    with pytest.raises(PresetError) as nothing:
        page.save_preset("mine", {"nope": ["a=1"]}, "")
    assert str(nothing.value) == "no plugin named nope; nothing saved"

    with pytest.raises(PresetError) as filled:
        page.save_preset("mine", {"scan-plan": ["wells=A1"]}, "")
    assert str(filled.value) == (
        "scan-plan: wells is not kept under params: the preset's wells hold them; "
        "nothing saved"
    )
    assert not (tmp_path / "profiles").exists()


def test_save_preset_refuses_a_bad_name_before_anything_else(
    make_page: Callable[..., PlatePage],
) -> None:
    with pytest.raises(PresetError, match=r"^a preset's name is 1 to 64"):
        make_page().save_preset("My Preset", {"scan-plan": ["coverage=lots"]}, "")


def test_the_saved_preset_becomes_the_loaded_one(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()
    page.select_text("B2,B11")

    saved = page.save_preset(
        "mine", {"scan-plan": ["coverage=0.4"], "cells": ["detector=blobs"]}, "B2,B11"
    )

    assert page.preset is saved
    assert saved.file.name == "mine"
    assert saved.file.path == tmp_path / "profiles" / "presets" / "mine.toml"
    assert saved.fields == {
        "scan-plan": {"coverage": "0.4"},
        "cells": {"detector": "blobs"},
    }
    assert saved.mismatches == ()
    # Loading it again changes nothing on the page.
    before = (page.plate, page.selection)
    page.load_preset("mine")
    assert (page.plate, page.selection) == before


def test_save_preset_refuses_an_existing_name_until_it_is_told_to_overwrite(
    make_page: Callable[..., PlatePage], tmp_path: Path
) -> None:
    page = make_page()
    page.save_preset("mine", {"cells": ["detector=blobs"]}, "")
    first = _saved(tmp_path, "mine")

    with pytest.raises(PresetExistsError) as exists:
        page.save_preset("mine", {"cells": ["detector=model"]}, "")

    assert exists.value.path == tmp_path / "profiles" / "presets" / "mine.toml"
    assert _saved(tmp_path, "mine") == first
    page.save_preset("mine", {"cells": ["detector=model"]}, "", overwrite=True)
    assert _saved(tmp_path, "mine")["params"] == {"cells": ["detector=model"]}


def test_a_preset_the_page_saved_loads_in_a_new_page_of_the_same_folders(
    make_page: Callable[..., PlatePage],
) -> None:
    first = make_page()
    first.select_text("A1:B2")
    first.save_preset("mine", {"scan-plan": ["coverage=0.4"]}, "A1:B2")

    second = make_page()
    loaded = second.load_preset("mine")

    assert second.selection == first.selection
    assert loaded.fields == {"scan-plan": {"coverage": "0.4"}}


# --- the next step ----------------------------------------------------------------

SUGGEST = {
    "scan-plan": {
        "params": ["coverage=0.4", "overlap=0.2"],
        "why": "fast on the demo stage",
    }
}


def _pairs_of(command: str) -> list[str]:
    words = shlex.split(command)
    return [words[i + 1] for i, word in enumerate(words) if word == "--param"]


def test_the_next_step_takes_the_preset_over_the_profile_key_by_key(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc)
    path = _put(tmp_path, text=WEEKLY.replace("coverage=0.4", "coverage=0.7"))
    page = make_page(_profile(SUGGEST))
    page.load_preset(str(path))

    step = page.view().next_step

    assert step is not None
    assert step.plugin == "scan-plan"
    # The page's wells, the profile's key the preset does not give, then the preset's.
    assert list(step.params) == ["wells=B2,B11", "overlap=0.2", "coverage=0.7"]
    assert step.why == (
        "the run document has no scan-plan yet",
        "wells: the 2 wells selected on the map",
        "overlap=0.2: suggested by the profile demo-sample: fast on the demo stage",
        "coverage=0.7: from the preset weekly",
    )


def test_a_suggestion_the_preset_overrides_whole_leaves_no_line_of_the_profile(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc)
    path = _put(
        tmp_path,
        text=WEEKLY.replace('["coverage=0.4"]', '["coverage=0.7", "overlap=0.3"]'),
    )
    page = make_page(_profile(SUGGEST))
    page.load_preset(str(path))

    step = page.view().next_step

    assert step is not None
    assert list(step.params) == ["wells=B2,B11", "coverage=0.7", "overlap=0.3"]
    assert step.why == (
        "the run document has no scan-plan yet",
        "wells: the 2 wells selected on the map",
        "coverage=0.7 overlap=0.3: from the preset weekly",
    )


def test_a_tool_the_preset_does_not_name_keeps_the_profiles_suggestion_alone(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc)
    path = _put(tmp_path, text='stand = "demo-sample"\nwells = "A1"\n')
    page = make_page(_profile(SUGGEST))
    page.load_preset(str(path))

    step = page.view().next_step

    assert step is not None
    assert list(step.params) == ["wells=A1", "coverage=0.4", "overlap=0.2"]
    assert step.why[-1] == (
        "coverage=0.4 overlap=0.2: suggested by the profile demo-sample: "
        "fast on the demo stage"
    )
    assert not any("from the preset" in line for line in step.why)


def test_the_next_step_command_names_the_preset(
    make_page: Callable[..., PlatePage], tmp_path: Path, run_doc: Path
) -> None:
    _plate_done(run_doc)
    path = _put(tmp_path)
    page = make_page()
    page.load_preset(str(path))

    step = page.view().next_step

    assert step is not None
    words = shlex.split(step.command)
    assert words[words.index("--preset") + 1] == str(path)
    # The pairs repeat what the preset gives: the command says everything it runs.
    assert _pairs_of(step.command) == list(step.params)
    # And without a preset the command has none.
    page.forget_preset()
    again = page.view().next_step
    assert again is not None
    assert "--preset" not in shlex.split(again.command)
