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
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
from pymmcore_plus import CMMCorePlus
from typer.testing import CliRunner, Result

import smc.cli as cli
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.profile import Profile
from smc.plugins import registry as registry_mod
from smc.plugins.api import PluginParams, PluginResult, RunContext
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
def test_presets_help_lists_list_and_show(env: dict[str, str] | None) -> None:
    # FM-72: typer forces colour on GitHub Actions, so the text is compared plain.
    result = _run("presets", "--help", env=env)

    assert result.exit_code == 0, result.output
    plain = _plain(result)
    assert re.search(r"\blist\b", plain)
    assert re.search(r"\bshow\b", plain)


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
