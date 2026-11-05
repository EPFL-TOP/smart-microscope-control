"""``smc run --preset`` and ``smc presets list|show`` (m5 §10.3, #319).

The runner pins the width (FM-37) and clears ``SMC_PROFILE`` and ``SMC_PROFILES``,
so an operator's environment never decides where a preset is looked for. Every
test works in ``tmp_path`` as the working directory, so that the folders a preset
is searched in (``./profiles/presets``, ``./presets``) are the test's own and none
writes into the repository. The test plugins are classes of this module, put in
the registry as ``test_cli_run`` does; the checks that need no stand run under
``no_stand``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pymmcore_plus import CMMCorePlus
from typer.testing import CliRunner, Result

import smc.cli as cli
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins import registry as registry_mod
from smc.plugins.api import PluginParams, PluginResult, RunContext, StopRequested
from smc.plugins.registry import GROUP
from smc.plugins.scan_plan.plugin import ScanPlanParams
from smc.plugins.slots import PlateResult
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.testing import FakeCore

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

THIS = __name__
REPO = Path(__file__).resolve().parents[1]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PANEL = re.compile(r"[│╭╮╰╯─]")


# -- plugins of this module ------------------------------------------------------


class PlanParams(PluginParams):
    wells: str = ""
    coverage: float = 1.0


class CellsParams(PluginParams):
    detector: str = "model"


class PlateParams(PluginParams):
    plate: str = ""
    refs: str = ""


class EchoResult(PluginResult):
    wells: str = ""
    coverage: float = 0.0
    detector: str = ""
    plate: str = ""


class Plan:
    name = provides = "scan-plan"
    summary = "Returns the wells and the coverage it was given."
    Params = PlanParams
    Result = EchoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: PlanParams, ctx: RunContext) -> EchoResult:
        return EchoResult(wells=params.wells, coverage=params.coverage)


class Cells:
    name = provides = "cells"
    summary = "Returns the detector it was given."
    Params = CellsParams
    Result = EchoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: CellsParams, ctx: RunContext) -> EchoResult:
        return EchoResult(detector=params.detector)


class Registration:
    name = "plate-registration"
    provides = "plate"
    summary = "Returns the plate it was given."
    Params = PlateParams
    Result = EchoResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: PlateParams, ctx: RunContext) -> EchoResult:
        return EchoResult(plate=params.plate)


class Seen(BaseModel):
    done: int


class StoppingPlan(Plan):
    """``scan-plan`` that stops itself after one unit; the attempt resumed from it finishes."""

    def run(self, microscope: Any, params: PlanParams, ctx: RunContext) -> EchoResult:
        if ctx.checkpoint(Seen) is None:
            ctx.save_checkpoint(Seen(done=1))
            raise StopRequested
        return EchoResult(wells=params.wells, coverage=params.coverage)


def _register(monkeypatch: pytest.MonkeyPatch, *classes: type) -> None:
    entries = [
        EntryPoint(name=cls.name, value=f"{THIS}:{cls.__name__}", group=GROUP)  # type: ignore[attr-defined]
        for cls in classes
    ]
    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: list(entries))


# -- fixtures --------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The working directory is the test's: the preset folders are ``./presets`` and ``./profiles/presets``."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def no_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command opens a microscope."""

    def refuse(cls: type[Microscope], *args: object, **kwargs: object) -> Microscope:
        pytest.fail("the command opened a microscope")

    monkeypatch.setattr(Microscope, "open", classmethod(refuse))


@pytest.fixture
def fake_stand(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """A ``FakeCore`` stand built from the profile the command passes; the list holds each ``dry_run``."""
    opened: list[bool] = []

    def open_stand(
        cls: type[Microscope], profile: object = "demo", *, dry_run: bool = False
    ) -> Microscope:
        opened.append(dry_run)
        built = profile if isinstance(profile, Profile) else Profile.demo()
        return Microscope.from_core(
            cast("CMMCorePlus", FakeCore.demo_like()), built, dry_run=dry_run
        )

    monkeypatch.setattr(Microscope, "open", classmethod(open_stand))
    return opened


@pytest.fixture
def demo(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


def _put(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


WEEKLY = """\
stand = "demo"
plate = "96-well"
wells = "B2,B11"

[params]
scan-plan = ["coverage=0.4"]
cells = ["detector=blobs"]
"""


def _weekly(tmp_path: Path, text: str = WEEKLY) -> Path:
    return _put(tmp_path / "presets", "weekly", text)


def _run(*args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, list(args), env=env)


def _doc(tmp_path: Path) -> str:
    return str(tmp_path / "run.json")


def _lines(result: Result) -> list[str]:
    return result.output.splitlines()


def _plain(result: Result) -> str:
    """The output without colour codes, box lines or runs of blanks (FM-72)."""
    return " ".join(_PANEL.sub(" ", _ANSI.sub("", result.output)).split())


def _json_in(output: str) -> Any:
    lines = output.splitlines()
    value, _ = json.JSONDecoder().raw_decode("\n".join(lines[lines.index("{") :]))
    return value


def _stored(tmp_path: Path, plugin: str) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages[plugin]


def _register_plate(tmp_path: Path, plate: str, *, profile: str = "demo") -> None:
    """A ``done`` plate record, as plate-registration would have left it."""
    then = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
    result = PlateResult(
        plate=plate,
        a1_center_xy_um=(1000.0, -500.0),
        rotation_deg=0.0,
        method="seed",
        residual_um=None,
        scale_error=None,
        n_refs=0,
    )
    write_stage(
        tmp_path / "run.json",
        StageRecord(
            plugin="plate-registration",
            provides="plate",
            attempt="a1",
            status=StageStatus.done,
            started_utc=then,
            finished_utc=then,
            profile=profile,
            dry_run=False,
            smc_version="0",
            params={},
            result=result.model_dump(mode="json"),
        ),
    )


# -- smc run --preset ------------------------------------------------------------


def test_run_with_preset_applies_its_params_and_the_record_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Plan)
    path = _weekly(tmp_path)

    result = _run(
        "run", "scan-plan", "--preset", str(path), "--run-doc", _doc(tmp_path)
    )

    assert result.exit_code == 0, result.output
    assert _json_in(result.output)["wells"] == "B2,B11"
    assert _json_in(result.output)["coverage"] == 0.4
    record = _stored(tmp_path, "scan-plan")
    assert record.preset == "weekly"
    assert record.params == {"wells": "B2,B11", "coverage": 0.4}
    # The preset also holds `cells`, a plugin this run does not load: not a mismatch.
    assert "preset weekly" not in result.output


def test_a_preset_is_found_by_its_name_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Cells)
    _weekly(tmp_path)

    result = _run("run", "cells", "--preset", "weekly", "--run-doc", _doc(tmp_path))

    assert result.exit_code == 0, result.output
    assert _json_in(result.output)["detector"] == "blobs"
    assert _stored(tmp_path, "cells").preset == "weekly"


def test_param_wins_over_the_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Plan)
    path = _weekly(tmp_path)

    result = _run(
        "run",
        "scan-plan",
        "--preset",
        str(path),
        "--param",
        "coverage=0.5",
        "--param",
        "wells=A1",
        "--run-doc",
        _doc(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert _stored(tmp_path, "scan-plan").params == {"wells": "A1", "coverage": 0.5}
    assert _stored(tmp_path, "scan-plan").preset == "weekly"


def test_a_run_without_preset_records_none_and_writes_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Plan)

    result = _run("run", "scan-plan", "--run-doc", _doc(tmp_path))

    assert result.exit_code == 0, result.output
    stored = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "preset" not in stored["stages"]["scan-plan"]


def test_run_with_preset_prints_a_parameter_the_plugin_no_longer_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Plan)
    path = _weekly(
        tmp_path, WEEKLY.replace('["coverage=0.4"]', '["coverage=0.4", "overlap=0.2"]')
    )

    result = _run(
        "run", "scan-plan", "--preset", str(path), "--run-doc", _doc(tmp_path)
    )

    assert result.exit_code == 0, result.output
    assert (
        "preset weekly [params.scan-plan]: overlap is not a parameter of scan-plan "
        "(keys: wells, coverage)"
    ) in _lines(result)
    # Only its own pair is dropped: the rest of the assay runs.
    assert _stored(tmp_path, "scan-plan").params == {"wells": "B2,B11", "coverage": 0.4}


def test_run_with_preset_prints_the_plate_mismatch_and_does_not_apply_its_wells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Plan)
    path = _weekly(tmp_path)
    _register_plate(tmp_path, "384-well")

    result = _run(
        "run", "scan-plan", "--preset", str(path), "--run-doc", _doc(tmp_path)
    )

    assert result.exit_code == 0, result.output
    assert (
        "preset weekly is for a 96-well plate; this run document's plate is "
        "registered as a 384-well: its plate and wells are not applied"
    ) in _lines(result)
    record = _stored(tmp_path, "scan-plan")
    assert record.params == {"wells": "", "coverage": 0.4}
    assert record.preset == "weekly"


def test_run_with_preset_names_another_stand_and_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Cells)
    path = _weekly(tmp_path, WEEKLY.replace('stand = "demo"', 'stand = "nikon-ti2"'))

    result = _run("run", "cells", "--preset", str(path), "--run-doc", _doc(tmp_path))

    assert result.exit_code == 0, result.output
    assert "preset weekly was saved on nikon-ti2; this stand is demo" in _lines(result)
    assert _stored(tmp_path, "cells").params == {"detector": "blobs"}


def test_the_plate_of_a_preset_reaches_plate_registration_and_refs_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Registration)
    path = _weekly(
        tmp_path, WEEKLY + 'plate-registration = ["plate=6-well", "refs=[]"]\n'
    )

    result = _run(
        "run", "plate-registration", "--preset", str(path), "--run-doc", _doc(tmp_path)
    )

    assert result.exit_code == 0, result.output
    # The preset's plate is applied, and what it keeps under params is named and dropped.
    assert _stored(tmp_path, "plate-registration").params == {
        "plate": "96-well",
        "refs": "",
    }
    assert (
        "preset weekly [params.plate-registration]: plate is not kept under params: "
        "the preset's plate holds it"
    ) in _lines(result)
    assert (
        "preset weekly [params.plate-registration]: refs is not kept under params: "
        "a preset holds no positions"
    ) in _lines(result)


def test_a_bare_toml_file_name_is_a_path_not_a_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    # `NAME|PATH`: `mine.toml` in the working directory is read as the file it is, and its
    # name is its stem; it is in no preset folder, so a name would not find it.
    _register(monkeypatch, Cells)
    _put(tmp_path, "mine", WEEKLY)

    result = _run("run", "cells", "--preset", "mine.toml", "--run-doc", _doc(tmp_path))

    assert result.exit_code == 0, result.output
    assert _stored(tmp_path, "cells").preset == "mine"
    assert _stored(tmp_path, "cells").params == {"detector": "blobs"}


def test_a_resumed_stage_names_the_preset_and_a_changed_preset_is_the_guards_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    # The resume guard compares the parameters as it always did; no rule of its own for a
    # preset (the plan's "do not add a preset-specific rule").
    _register(monkeypatch, StoppingPlan)
    path = _weekly(tmp_path)
    doc = _doc(tmp_path)
    stopped = _run("run", "scan-plan", "--preset", str(path), "--run-doc", doc)
    assert stopped.exit_code == 130, stopped.output
    assert _stored(tmp_path, "scan-plan").status is StageStatus.stopped
    assert _stored(tmp_path, "scan-plan").preset == "weekly"

    path.write_text(WEEKLY.replace("coverage=0.4", "coverage=0.9"), encoding="utf-8")
    changed = _run(
        "run", "scan-plan", "--preset", str(path), "--resume", "--run-doc", doc
    )
    assert changed.exit_code == 2
    assert "the parameters differ from the run being resumed: coverage" in _plain(
        changed
    )
    assert _stored(tmp_path, "scan-plan").status is StageStatus.stopped

    path.write_text(WEEKLY, encoding="utf-8")
    resumed = _run(
        "run", "scan-plan", "--preset", str(path), "--resume", "--run-doc", doc
    )
    assert resumed.exit_code == 0, resumed.output
    record = _stored(tmp_path, "scan-plan")
    assert record.status is StageStatus.done
    assert record.preset == "weekly"


def test_an_unknown_preset_exits_2_and_names_the_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Plan)

    result = _run("run", "scan-plan", "--preset", "nope", "--run-doc", _doc(tmp_path))

    assert result.exit_code == 2
    plain = _plain(result)
    assert "Invalid value for --preset: no preset named nope; looked in" in plain
    assert "no preset named nope; looked in" in plain
    assert "smc presets list shows the presets" in plain
    assert not (tmp_path / "run.json").exists()


def test_an_invalid_preset_exits_2_and_names_its_file_and_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Plan)
    path = _weekly(tmp_path, 'colour = "red"\n')

    result = _run(
        "run", "scan-plan", "--preset", str(path), "--run-doc", _doc(tmp_path)
    )

    assert result.exit_code == 2
    assert "colour: unknown key; the keys are stand, plate, wells" in _plain(result)


def test_a_bad_param_with_a_preset_is_refused_with_the_plugin_and_opens_no_stand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Plan)
    path = _weekly(tmp_path)

    result = _run(
        "run",
        "scan-plan",
        "--preset",
        str(path),
        "--param",
        "coverage=lots",
        "--run-doc",
        _doc(tmp_path),
    )

    assert result.exit_code == 2
    assert "scan-plan" in _plain(result)
    assert "coverage" in _plain(result)
    assert not (tmp_path / "run.json").exists()


def test_a_preset_value_the_plugin_refuses_is_that_runs_params_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    # A value is not checked by the preset (§10.2): the run refuses it, as for [suggest].
    _register(monkeypatch, Plan)
    path = _weekly(tmp_path, WEEKLY.replace("coverage=0.4", "coverage=lots"))

    result = _run(
        "run", "scan-plan", "--preset", str(path), "--run-doc", _doc(tmp_path)
    )

    assert result.exit_code == 2
    assert "coverage" in _plain(result)


# -- smc presets -----------------------------------------------------------------


def test_presets_list_shows_each_preset_and_a_broken_file(tmp_path: Path) -> None:
    good = _weekly(tmp_path)
    bad = _put(tmp_path / "presets", "bad", "wells = \n")
    other = _put(tmp_path / "profiles" / "presets", "alpha", "")

    result = _run("presets", "list")

    assert result.exit_code == 0, result.output
    lines = _lines(result)
    folders = f"{tmp_path / 'profiles' / 'presets'}, {tmp_path / 'presets'}"
    # Names are padded to the widest; `-` stands for what the preset leaves out.
    plugins = "scan-plan,cells"
    assert f"alpha   -     -        {'-'.ljust(len(plugins))}  {other}" in lines
    assert f"weekly  demo  96-well  scan-plan,cells  {good}" in lines
    broken = next(line for line in lines if line.startswith("bad     cannot be read: "))
    assert f"{bad}: not TOML: " in broken
    assert broken.endswith(f"  {bad}")
    assert lines[-1] == f"searched: {folders}"
    assert [line.split()[0] for line in lines[:-1]] == ["alpha", "bad", "weekly"]


def test_presets_list_looks_in_the_folder_of_the_profile_it_is_given(
    tmp_path: Path,
) -> None:
    # `-p` names the stand whose presets are listed: a preset beside a profile that is
    # outside the working directory is found, and its folder heads the search.
    stand = tmp_path / "stand"
    stand.mkdir()
    profile = stand / "bench.toml"
    profile.write_text('[microscope]\nname = "bench"\n', encoding="utf-8")
    own = _put(stand / "presets", "own", 'stand = "bench"\n')

    result = _run("presets", "list", "-p", str(profile))

    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert lines[0] == f"own  bench  -  -  {own}"
    assert lines[-1] == (
        f"searched: {stand / 'presets'}, {tmp_path / 'profiles' / 'presets'}, "
        f"{tmp_path / 'presets'}"
    )
    # Without -p it is the demo's folders, and the stand's preset is not there.
    assert _run("presets", "list").output.splitlines()[0].startswith("no presets;")


def test_presets_list_with_none_says_where_it_searched(tmp_path: Path) -> None:
    result = _run("presets", "list")

    assert result.exit_code == 0, result.output
    assert _lines(result) == [
        f"no presets; searched: {tmp_path / 'profiles' / 'presets'}, "
        f"{tmp_path / 'presets'}"
    ]


def test_presets_show_prints_the_pairs_and_the_mismatches(
    tmp_path: Path, no_stand: None
) -> None:
    path = _weekly(
        tmp_path,
        WEEKLY.replace('stand = "demo"', 'stand = "nikon-ti2"')
        .replace('"coverage=0.4"', '"coverage=0.4", "bogus=1"')
        .replace(
            "[params]",
            'saved_utc = 2026-11-03 10:00:00+00:00\n\n[params]\nnoplugin = ["a=1"]',
        ),
    )
    _register_plate(tmp_path, "384-well")

    result = _run("presets", "show", "weekly", "--run-doc", _doc(tmp_path))

    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert lines[:5] == [
        f"preset weekly  {path}",
        "stand: nikon-ti2",
        "plate: 96-well",
        "wells: B2,B11",
        "saved_utc: 2026-11-03T10:00:00+00:00",
    ]
    assert "noplugin: a=1" in lines
    assert "scan-plan: coverage=0.4 bogus=1" in lines
    assert "cells: detector=blobs" in lines
    keys = ", ".join(ScanPlanParams.model_fields)
    assert "preset weekly was saved on nikon-ti2; this stand is demo" in lines
    assert (
        "preset weekly is for a 96-well plate; this run document's plate is "
        "registered as a 384-well: its plate and wells are not applied"
    ) in lines
    assert "preset weekly [params.noplugin]: no plugin named noplugin" in lines
    assert (
        "preset weekly [params.scan-plan]: bogus is not a parameter of scan-plan "
        f"(keys: {keys})"
    ) in lines
    assert not any(line.startswith("no mismatch") for line in lines)


def test_presets_show_says_when_nothing_mismatches(
    tmp_path: Path, no_stand: None
) -> None:
    _weekly(tmp_path)

    result = _run("presets", "show", "weekly")

    assert result.exit_code == 0, result.output
    assert _lines(result)[-1] == "no mismatch on demo"


def test_presets_show_reads_a_missing_run_document_as_registering_nothing(
    tmp_path: Path, no_stand: None
) -> None:
    _weekly(tmp_path)

    result = _run("presets", "show", "weekly", "--run-doc", _doc(tmp_path))

    assert result.exit_code == 0, result.output
    assert _lines(result)[-1] == "no mismatch on demo"
    assert not (tmp_path / "run.json").exists()


def test_presets_show_of_an_unknown_preset_exits_2_with_one_line(
    tmp_path: Path, no_stand: None
) -> None:
    result = _run("presets", "show", "nope")

    assert result.exit_code == 2
    assert _lines(result) == [
        f"✗ no preset named nope; looked in {tmp_path / 'profiles' / 'presets'}, "
        f"{tmp_path / 'presets'}; smc presets list shows the presets"
    ]


def test_the_weekly_example_loads_on_demo_sample_without_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    # The checked-in example, read from the repository's own folders.
    monkeypatch.chdir(REPO)

    result = _run("presets", "show", "weekly", "-p", "demo-sample")

    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert lines[0] == f"preset weekly  {REPO / 'profiles' / 'presets' / 'weekly.toml'}"
    assert lines[-1] == "no mismatch on demo-sample"


@pytest.mark.parametrize("env", [None, {"GITHUB_ACTIONS": "1"}])
def test_presets_help_lists_list_show_and_save(env: dict[str, str] | None) -> None:
    # FM-72: typer forces colour on GitHub Actions, so the text is compared plain.
    result = _run("presets", "--help", env=env)

    assert result.exit_code == 0, result.output
    plain = _plain(result)
    assert re.search(r"\blist\b", plain)
    assert re.search(r"\bshow\b", plain)
    assert re.search(r"\bsave\b", plain)


@pytest.mark.parametrize("env", [None, {"GITHUB_ACTIONS": "1"}])
def test_run_help_declares_the_preset_option(env: dict[str, str] | None) -> None:
    result = _run("run", "--help", env=env)

    assert result.exit_code == 0, result.output
    assert re.search(r"--preset\b", _plain(result))


def test_presets_is_a_command_of_smc() -> None:
    assert _run("--help").exit_code == 0
    assert "presets" in _plain(_run("--help"))
    assert cli.presets_app is not None


# -- the acceptance: demo-sample -----------------------------------------------


@pytest.mark.usefixtures("demo")
def test_the_weekly_preset_runs_scan_plan_on_demo_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `smc run scan-plan -p demo-sample --preset weekly --run-doc run.json`, on a
    # document whose plate is registered (m5 §10, #319).
    monkeypatch.chdir(REPO)
    document = tmp_path / "run.json"
    _register_plate(tmp_path, "96-well", profile="demo-sample")

    first = _run(
        "run",
        "scan-plan",
        "-p",
        "demo-sample",
        "--preset",
        "weekly",
        "--run-doc",
        str(document),
    )

    assert first.exit_code == 0, first.output
    record = RunDocument.load(document).stages["scan-plan"]
    assert record.params["wells"] == "B2,B11"
    assert record.params["coverage"] == 0.4
    assert record.preset == "weekly"

    second = _run(
        "run",
        "scan-plan",
        "-p",
        "demo-sample",
        "--preset",
        "weekly",
        "--param",
        "coverage=0.5",
        "--run-doc",
        str(document),
    )

    assert second.exit_code == 0, second.output
    again = RunDocument.load(document).stages["scan-plan"]
    assert again.params["coverage"] == 0.5
    assert again.params["wells"] == "B2,B11"
    assert again.preset == "weekly"


# -- smc presets save (m5 §10.9, #324) -----------------------------------------------


def _stage(
    tmp_path: Path,
    plugin: str,
    params: dict[str, Any],
    *,
    provides: str | None = None,
    status: StageStatus = StageStatus.done,
    profile: str = "demo",
    document: str = "a.json",
) -> Path:
    """A record as the runner leaves it, in ``tmp_path/document``; the file is returned."""
    then = datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc)
    write_stage(
        tmp_path / document,
        StageRecord(
            plugin=plugin,
            provides=provides or plugin,
            attempt="a1",
            status=status,
            started_utc=then,
            finished_utc=None if status is StageStatus.running else then,
            profile=profile,
            dry_run=False,
            smc_version="0",
            params=params,
        ),
    )
    return tmp_path / document


def _saved(tmp_path: Path) -> Path:
    """Where a save on the demo stand writes ``weekly``: the first folder searched."""
    return tmp_path / "profiles" / "presets" / "weekly.toml"


def test_presets_save_writes_a_preset_that_run_preset_reproduces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stand: list[bool]
) -> None:
    _register(monkeypatch, Plan, Cells, Registration)
    first = str(tmp_path / "a.json")
    for stage in (
        ("plate-registration", "plate=96-well"),
        ("scan-plan", "wells=B2,B11", "coverage=0.4"),
        ("cells", "detector=blobs"),
    ):
        args = ["run", stage[0], "--run-doc", first]
        for pair in stage[1:]:
            args += ["--param", pair]
        assert _run(*args).exit_code == 0

    saved = _run("presets", "save", "weekly", "--run-doc", first)

    assert saved.exit_code == 0, saved.output
    assert _saved(tmp_path).is_file()
    _register_plate(tmp_path, "96-well")
    again = _run("run", "scan-plan", "--preset", "weekly", "--run-doc", _doc(tmp_path))
    assert again.exit_code == 0, again.output
    recorded = RunDocument.load(tmp_path / "a.json").stages["scan-plan"]
    reproduced = _stored(tmp_path, "scan-plan")
    assert reproduced.params == recorded.params == {"wells": "B2,B11", "coverage": 0.4}
    assert reproduced.preset == "weekly"
    shown = _run("presets", "show", "weekly")
    assert _lines(shown)[-1] == "no mismatch on demo"


@pytest.mark.usefixtures("demo")
def test_the_demo_sample_chain_is_saved_and_reproduced(tmp_path: Path) -> None:
    # The acceptance of #324 on the real chain. The profile is a copy in `tmp_path`:
    # a preset is written beside its profile, and the repository's own
    # `profiles/presets` must stay as it is.
    checked_in = REPO / "profiles" / "presets" / "weekly.toml"
    untouched = checked_in.read_bytes()
    profile = tmp_path / "demo-sample.toml"
    profile.write_text(
        (REPO / "profiles" / "demo-sample.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    refs = (
        '[{"well":"A1","x_um":-49500,"y_um":31500},'
        '{"well":"A12","x_um":49500,"y_um":31500},'
        '{"well":"H1","x_um":-49500,"y_um":-31500}]'
    )
    first = str(tmp_path / "a.json")

    def stage(name: str, *pairs: str, document: str = first, preset: str = "") -> None:
        args = ["run", name, "-p", str(profile), "--run-doc", document]
        args += ["--preset", preset] if preset else []
        for pair in pairs:
            args += ["--param", pair]
        result = _run(*args)
        assert result.exit_code == 0, f"{name}: {result.output}"

    stage("plate-registration", f"refs={refs}")
    stage("scan-plan", "wells=B2,B11", "coverage=0.4")

    saved = _run("presets", "save", "weekly", "--run-doc", first, "-p", str(profile))

    assert saved.exit_code == 0, saved.output
    assert (tmp_path / "presets" / "weekly.toml").is_file()
    assert checked_in.read_bytes() == untouched
    _register_plate(tmp_path, "96-well", profile="demo-sample")
    stage("scan-plan", document=_doc(tmp_path), preset="weekly")
    recorded = RunDocument.load(tmp_path / "a.json").stages["scan-plan"]
    reproduced = _stored(tmp_path, "scan-plan")
    assert reproduced.params == recorded.params
    assert reproduced.preset == "weekly"


def test_presets_save_refuses_an_existing_name_without_overwrite_and_overwrite_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Plan)
    first = _stage(tmp_path, "scan-plan", {"wells": "B2", "coverage": 0.4})
    second = _stage(
        tmp_path, "scan-plan", {"wells": "B2", "coverage": 0.5}, document="b.json"
    )
    assert _run("presets", "save", "weekly", "--run-doc", str(first)).exit_code == 0
    before = _saved(tmp_path).read_bytes()

    refused = _run("presets", "save", "weekly", "--run-doc", str(second))

    assert refused.exit_code == 2
    assert _lines(refused) == [
        f"✗ preset weekly exists: {_saved(tmp_path)}; pass --overwrite to replace it"
    ]
    assert _saved(tmp_path).read_bytes() == before

    replaced = _run(
        "presets", "save", "weekly", "--run-doc", str(second), "--overwrite"
    )

    assert replaced.exit_code == 0, replaced.output
    assert _saved(tmp_path).read_bytes() != before
    assert "coverage=0.5" in _saved(tmp_path).read_text(encoding="utf-8")


def test_presets_save_refuses_a_bad_name_before_it_reads_the_document(
    tmp_path: Path, no_stand: None
) -> None:
    result = _run(
        "presets", "save", "Weekly", "--run-doc", str(tmp_path / "nowhere.json")
    )

    assert result.exit_code == 2
    plain = _plain(result)
    assert (
        "Invalid value for NAME: a preset's name is 1 to 64 lower-case letters, "
        "digits, - and _, starting with a letter or a digit; got 'Weekly'"
    ) in plain
    assert "does not exist" not in plain
    assert not (tmp_path / "profiles").exists()


def test_presets_save_of_a_missing_run_document_exits_1_and_writes_nothing(
    tmp_path: Path, no_stand: None
) -> None:
    missing = tmp_path / "nowhere.json"

    result = _run("presets", "save", "weekly", "--run-doc", str(missing))

    assert result.exit_code == 1
    assert _lines(result) == [f"✗ {missing}: does not exist"]
    assert not (tmp_path / "profiles").exists()


def test_presets_save_of_a_document_with_nothing_finished_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Plan)
    path = _stage(tmp_path, "scan-plan", {"wells": "B2"}, status=StageStatus.failed)

    result = _run("presets", "save", "weekly", "--run-doc", str(path))

    assert result.exit_code == 2
    assert _lines(result) == [
        "✗ the run document has no finished stage; there is nothing to save"
    ]
    assert not (tmp_path / "profiles").exists()


def test_presets_save_prints_the_saved_line_the_preset_and_the_notes_in_that_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Plan, Cells)
    _stage(tmp_path, "scan-plan", {"wells": "B2", "coverage": 0.4}, profile="nikon-ti2")
    path = _stage(tmp_path, "cells", {"detector": "blobs"})

    result = _run("presets", "save", "weekly", "--run-doc", str(path))

    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert lines[0] == f"saved weekly to {_saved(tmp_path)}"
    assert lines[1:4] == ["stand: demo", "plate: -", "wells: B2"]
    # To the second, in UTC: the page's own `saved_utc`.
    assert re.fullmatch(r"saved_utc: \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", lines[4])
    assert lines[5:] == [
        "cells: detector=blobs",
        "scan-plan: coverage=0.4",
        "scan-plan was run on nikon-ti2, not on demo",
    ]


def test_presets_save_writes_the_stand_of_the_profile_it_is_given_beside_that_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    # `-p` is a path whose text is not the stand's name: the preset holds the name.
    _register(monkeypatch, Cells)
    folder = tmp_path / "bench"
    folder.mkdir()
    profile = folder / "bench.toml"
    profile.write_text('[microscope]\nname = "bench-ti2"\n', encoding="utf-8")
    path = _stage(tmp_path, "cells", {"detector": "blobs"}, profile="bench-ti2")

    result = _run(
        "presets", "save", "weekly", "--run-doc", str(path), "-p", str(profile)
    )

    written = folder / "presets" / "weekly.toml"
    assert result.exit_code == 0, result.output
    assert _lines(result)[:2] == [f"saved weekly to {written}", "stand: bench-ti2"]
    assert 'stand = "bench-ti2"' in written.read_text(encoding="utf-8")
    # The stage was run on that stand: no note.
    assert _lines(result)[-1] == "cells: detector=blobs"


def test_presets_save_reports_a_folder_it_cannot_write_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Cells)
    path = _stage(tmp_path, "cells", {"detector": "blobs"})
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "presets").write_text("a file, not a folder", "utf-8")

    result = _run("presets", "save", "weekly", "--run-doc", str(path))

    assert result.exit_code == 1
    lines = _lines(result)
    assert len(lines) == 1
    assert lines[0].startswith(f"✗ {_saved(tmp_path)}: not saved: ")


def test_presets_save_without_tomli_w_says_what_to_install_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stand: None
) -> None:
    _register(monkeypatch, Cells)
    path = _stage(tmp_path, "cells", {"detector": "blobs"})
    monkeypatch.setitem(sys.modules, "tomli_w", None)

    result = _run("presets", "save", "weekly", "--run-doc", str(path))

    assert result.exit_code == 1
    lines = _lines(result)
    assert len(lines) == 1
    assert lines[0].startswith("✗ saving a preset needs the tomli-w package")
    assert not _saved(tmp_path).exists()


@pytest.mark.parametrize("env", [None, {"GITHUB_ACTIONS": "1"}])
def test_presets_save_needs_run_doc_and_the_help_names_the_command_and_its_options(
    env: dict[str, str] | None,
) -> None:
    # FM-72: typer forces colour on GitHub Actions, so the text is compared plain.
    missing = _run("presets", "save", "weekly", env=env)
    group = _run("presets", "--help", env=env)
    command = _run("presets", "save", "--help", env=env)

    assert missing.exit_code == 2
    assert re.search(r"Missing option\s+'--run-doc'", _plain(missing))
    assert group.exit_code == 0, group.output
    assert re.search(r"\bsave\b", _plain(group))
    assert command.exit_code == 0, command.output
    assert re.search(r"--run-doc\b", _plain(command))
    assert re.search(r"--overwrite\b", _plain(command))
