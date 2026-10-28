"""``smc run`` (m3 §1.10): listing, parameters, a run, and every way it refuses.

The runner pins the width (FM-37) and clears ``SMC_PROFILE``, so an operator's
``SMC_PROFILE=nikon-ti2`` never makes ``pytest`` open a real stand (FM-40). The
checks that need no stand run under ``no_stand``, which fails the test if one is
opened (FM-30); the runs that open one use ``fake_stand`` (``FakeCore``, no
Micro-Manager) or, where the demo devices are the point, ``demo``. Test plugins are
classes of this module, put in the registry through ``entry_points`` (a monkeypatch
of ``smc.plugins.registry.entry_points``).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pymmcore_plus import CMMCorePlus
from typer.testing import CliRunner, Result

import smc
import smc.cli as cli
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.hardware.errors import HardwareError, SafetyRefusedError
from smc.hardware.profile import Profile
from smc.plugins import registry as registry_mod
from smc.plugins.api import PluginParams, PluginResult, RunContext, StopRequested
from smc.plugins.errors import PluginRunError
from smc.plugins.registry import GROUP
from smc.run import RunDocument, StageRecord, StageStatus, write_stage
from smc.run import files as files_mod
from smc.testing import FakeCore

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

THIS = __name__


# --- plugins of this module --------------------------------------------------


class Params(PluginParams):
    n: int = 1
    label: str = "x"
    wells: tuple[str, ...] = ()


class EchoResult(PluginResult):
    value: int = 0
    label: str = ""
    wells: tuple[str, ...] = ()


class Seen(BaseModel):
    done: int


class Base:
    name = "base"
    summary = "A test plugin."
    Params = Params
    Result = EchoResult
    requires: tuple[type, ...] = ()
    provides = "base"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        return EchoResult(value=params.n, label=params.label, wells=params.wells)


class Good(Base):
    name = provides = "good"
    summary = "Returns its parameters."


class Up(Base):
    name = provides = "up"
    summary = "Produces the slot up."


class Reader(Base):
    name = provides = "reader"
    summary = "Reads up, and timings when there are some."
    reads = ("up",)
    reads_optional = ("timings",)


class Third(Base):
    name = provides = "third"
    summary = "Reads reader, which reads up."
    reads = ("reader",)


class NeedsXY(Base):
    name = provides = "needs-xy"
    summary = "Needs the XY stage."
    requires = (XYStage,)


class NoParams(PluginParams):
    pass


class Bare(Base):
    name = provides = "bare"
    summary = "Takes no parameter."
    Params = NoParams

    def run(self, microscope: Any, params: NoParams, ctx: RunContext) -> EchoResult:
        return EchoResult()


class NoSummary(Base):
    name = provides = "no-summary"
    summary = None  # type: ignore[assignment]


class Boom(Base):
    name = provides = "boom"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        raise HardwareError("the stage did not answer")


class Unusable(Base):
    name = provides = "unusable"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        raise PluginRunError(self.name, "no plate found; run it again")


class Refuses(Base):
    name = provides = "refuses"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        raise SafetyRefusedError("the jog is too long")


class CtrlC(Base):
    name = provides = "ctrl-c"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        raise KeyboardInterrupt


class SelfStop(Base):
    name = provides = "self-stop"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        raise StopRequested


class Progressing(Base):
    name = provides = "progressing"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        for done in range(1, 6):
            ctx.progress(done, 5, "step")
        ctx.progress(6, None, "")
        return EchoResult(value=5)


class Resumable(Base):
    """Stops itself after one unit; the next attempt, resumed, finishes."""

    name = provides = "resumable"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        saved = ctx.checkpoint(Seen)
        if saved is None:
            ctx.save_checkpoint(Seen(done=1))
            raise StopRequested
        return EchoResult(value=saved.done + 1)


class Surrogate(Base):
    name = provides = "surrogate"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        return EchoResult(label=f"{chr(0xDC80)} µm")


def _entry(cls: type) -> EntryPoint:
    return EntryPoint(name=cls.name, value=f"{THIS}:{cls.__name__}", group=GROUP)  # type: ignore[attr-defined]


def _register(
    monkeypatch: pytest.MonkeyPatch,
    *classes: type,
    installed: bool = False,
    extra: tuple[EntryPoint, ...] = (),
) -> None:
    """Put these classes (and the installed plugins, on request) in the registry."""
    found = list(importlib.metadata.entry_points(group=GROUP)) if installed else []
    entries = [*found, *map(_entry, classes), *extra]
    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: list(entries))


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_smc_logging() -> Iterator[None]:
    # The root callback replaces the smc logger's handler and level.
    logger = logging.getLogger("smc")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    logger.propagate = saved[2]


@pytest.fixture
def demo(mm_available: bool) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")


@pytest.fixture
def no_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command opens a microscope."""

    def refuse(cls: type[Microscope], *args: object, **kwargs: object) -> Microscope:
        pytest.fail("the command opened a microscope")

    monkeypatch.setattr(Microscope, "open", classmethod(refuse))


@pytest.fixture
def fake_stand(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Open a ``FakeCore`` stand; the list holds the ``dry_run`` of each open.

    The stand is built from the profile it is given, as ``Microscope.open`` would:
    a fake that always answered as the demo would hide a command that passes the
    wrong profile (the stand's name is what a stage records).
    """
    opened: list[bool] = []

    def open_stand(
        cls: type[Microscope], profile: object = "demo", *, dry_run: bool = False
    ) -> Microscope:
        opened.append(dry_run)
        core = FakeCore.demo_like()
        built = profile if isinstance(profile, Profile) else Profile.demo()
        return Microscope.from_core(cast("CMMCorePlus", core), built, dry_run=dry_run)

    monkeypatch.setattr(Microscope, "open", classmethod(open_stand))
    return opened


@pytest.fixture
def stand_events(fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What happened, in order: the stand's release and every line the CLI printed."""
    events: list[str] = []
    real_close = Microscope.close
    real_print = cli.console.print

    def close(self: Microscope) -> None:
        events.append("stand closed")
        real_close(self)

    def record(*args: Any, **kwargs: Any) -> None:
        events.append(str(args[0]) if args else "")
        real_print(*args, **kwargs)

    monkeypatch.setattr(Microscope, "close", close)
    monkeypatch.setattr(cli.console, "print", record)
    return events


@pytest.fixture
def noxy_profile(tmp_path: Path) -> Path:
    """The demo devices with the XY stage excluded, so the role stays unfilled."""
    path = tmp_path / "noxy.toml"
    path.write_text(
        '[microscope]\nname = "noxy"\n\n[roles.exclude]\nxy_stage = ["xy"]\n',
        encoding="utf-8",
    )
    return path


def _run(*args: str) -> Result:
    return runner.invoke(app, list(args))


def _doc(tmp_path: Path) -> str:
    return str(tmp_path / "run.json")


def _lines(result: Result) -> list[str]:
    return result.output.splitlines()


def _json_in(output: str) -> Any:
    """The JSON object a run printed, wherever it starts."""
    lines = output.splitlines()
    value, _ = json.JSONDecoder().raw_decode("\n".join(lines[lines.index("{") :]))
    return value


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


_PANEL = re.compile(r"[│╭╮╰╯─]")


def _plain(result: Result) -> str:
    """The output without colour codes, box lines or runs of blanks.

    Typer draws its own usage-error panel, and CI forces its colour (FM-72): an
    option name inside a phrase is then styled apart from the words around it. The
    panel also wraps a message at the console's width, so a long temporary path
    would split a phrase: assert phrases that do not hold one.
    """
    return " ".join(_PANEL.sub(" ", _ANSI.sub("", result.output)).split())


def _stored(tmp_path: Path, name: str) -> Any:
    return RunDocument.load(tmp_path / "run.json").stages[name]


# --- smc run --list ----------------------------------------------------------


@pytest.mark.usefixtures("no_stand")
def test_run_list_shows_stage_report_without_opening_a_stand() -> None:
    result = _run("run", "--list")
    assert result.exit_code == 0, result.output
    assert any(
        re.fullmatch(
            r"stage-report\s+Read where the stage is, and send no move\.\s+"
            r"requires XYStage\s+provides stage-report\s+reads -",
            line,
        )
        for line in _lines(result)
    ), result.output


@pytest.mark.usefixtures("no_stand")
def test_run_list_shows_requires_provides_and_reads_with_optional_in_parentheses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(monkeypatch, Good, NeedsXY, Reader)
    result = _run("run", "--list")
    assert result.exit_code == 0, result.output
    rows = [re.split(r"\s{2,}", line) for line in _lines(result)]
    assert rows == [
        ["good", "Returns its parameters.", "requires -", "provides good", "reads -"],
        [
            "needs-xy",
            "Needs the XY stage.",
            "requires XYStage",
            "provides needs-xy",
            "reads -",
        ],
        [
            "reader",
            "Reads up, and timings when there are some.",
            "requires -",
            "provides reader",
            "reads up, (timings)",
        ],
    ]


@pytest.mark.usefixtures("no_stand")
def test_run_list_prints_a_broken_plugin_and_exits_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A plugin that does not load is a finding, not the end of the listing.
    ghost = EntryPoint(
        name="ghost", value="no_such_module_for_smc_cli_tests:Thing", group=GROUP
    )
    _register(monkeypatch, Good, extra=(ghost,))
    result = _run("run", "--list")
    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert lines[0].startswith("good  ")
    assert lines[1] == (
        "✗ ghost (no_such_module_for_smc_cli_tests:Thing): could not be imported: "
        "ModuleNotFoundError: No module named 'no_such_module_for_smc_cli_tests'"
    )


@pytest.mark.usefixtures("no_stand")
def test_run_list_with_no_plugin_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _register(monkeypatch)
    result = _run("run", "--list")
    assert result.exit_code == 0, result.output
    assert _lines(result) == [
        "no plugin is registered in the smc.plugins group; if smc was updated "
        "without reinstalling, run pip install -e . in the smc folder (plugins "
        "are registered when it is installed)"
    ]


# --- smc run NAME --show-params ----------------------------------------------


@pytest.mark.usefixtures("no_stand")
def test_show_params_lists_the_fields() -> None:
    result = _run("run", "stage-report", "--show-params")
    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert len(lines) == 3
    assert re.fullmatch(
        r"samples\s+int\s+1\s+ge=1 le=100  How many readings to take\.", lines[0]
    )
    assert re.fullmatch(
        r"interval_ms\s+float\s+0\.0\s+ge=0\.0 le=10000\.0  Wait between two "
        r"readings\.",
        lines[1],
    )
    assert re.fullmatch(
        r"include_z\s+bool\s+True\s+Also read Z when there is a focus stage\.",
        lines[2],
    )


@pytest.mark.usefixtures("no_stand")
def test_show_params_of_a_plugin_without_fields_says_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(monkeypatch, Bare)
    result = _run("run", "bare", "--show-params")
    assert result.exit_code == 0, result.output
    assert _lines(result) == ["none"]


# --- usage errors and refusals before the stand opens ------------------------


@pytest.mark.usefixtures("no_stand")
def test_a_bad_param_exits_2_before_opening_the_stand(tmp_path: Path) -> None:
    result = _run(
        "run", "stage-report", "--run-doc", _doc(tmp_path), "--param", "samples=0"
    )
    assert result.exit_code == 2, result.output
    assert (
        "✗ parameters of stage-report: samples: Input should be greater than or "
        "equal to 1. smc run stage-report --show-params lists the keys"
    ) in result.output
    assert not (tmp_path / "run.json").exists()


@pytest.mark.usefixtures("no_stand")
def test_every_param_problem_is_printed_on_one_line(tmp_path: Path) -> None:
    result = _run(
        "run",
        "stage-report",
        "--run-doc",
        _doc(tmp_path),
        "--param",
        "nope=1",
        "--param",
        "samples=x",
        "--param",
        "samples=2",
    )
    assert result.exit_code == 2, result.output
    (line,) = [line for line in _lines(result) if line.startswith("✗ ")]
    assert line == (
        "✗ parameters of stage-report: unknown key 'nope' (keys: samples, "
        "interval_ms, include_z); 'samples' given twice; samples: Input should be a "
        "valid integer, unable to parse string as an integer. smc run stage-report "
        "--show-params lists the keys"
    )


@pytest.mark.usefixtures("no_stand")
def test_a_missing_run_doc_option_exits_2() -> None:
    result = _run("run", "stage-report")
    assert result.exit_code == 2, result.output
    assert "a run needs --run-doc" in _plain(result)
    assert "never created where you did not say" in _plain(result)


@pytest.mark.usefixtures("no_stand")
@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_run_doc_exits_2(value: str) -> None:
    # FM-78: Path("") is the current folder.
    result = _run("run", "stage-report", "--run-doc", value)
    assert result.exit_code == 2, result.output
    assert "is empty" in _plain(result)
    assert "would be the current folder" in _plain(result)


@pytest.mark.usefixtures("no_stand")
def test_a_run_doc_in_a_folder_that_does_not_exist_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "run.json"
    result = _run("run", "stage-report", "--run-doc", str(missing))
    assert result.exit_code == 2, result.output
    assert "does not exist" in _plain(result)
    assert "the folder" in _plain(result)


@pytest.mark.usefixtures("no_stand")
def test_a_run_doc_that_is_a_folder_exits_2(tmp_path: Path) -> None:
    result = _run("run", "stage-report", "--run-doc", str(tmp_path))
    assert result.exit_code == 2, result.output
    assert "is a folder; name the file to write" in _plain(result)


@pytest.mark.usefixtures("no_stand")
def test_an_unreadable_run_doc_exits_1_and_is_left_as_it_was(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(b'{"schema_version": 1, "stages": ')
    result = _run("run", "stage-report", "--run-doc", str(path))
    assert result.exit_code == 1, result.output
    assert f"✗ {path}: is not valid JSON: " in result.output
    assert path.read_bytes() == b'{"schema_version": 1, "stages": '


@pytest.mark.usefixtures("no_stand")
def test_a_name_is_needed_unless_listing() -> None:
    result = _run("run")
    assert result.exit_code == 2, result.output
    assert "smc run --list shows the plugins" in _plain(result)


@pytest.mark.usefixtures("no_stand")
def test_listing_with_a_name_is_a_usage_error() -> None:
    result = _run("run", "stage-report", "--list")
    assert result.exit_code == 2, result.output
    assert "--list takes no plugin name" in _plain(result)


@pytest.mark.usefixtures("no_stand")
def test_an_unknown_plugin_exits_2_and_lists_the_available_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Good, Up)
    result = _run("run", "nothing", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 2, result.output
    assert (
        "✗ no plugin named 'nothing' (available: good, up); smc run --list shows them"
    ) in result.output


@pytest.mark.usefixtures("no_stand")
def test_a_plugin_that_does_not_load_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ghost = EntryPoint(
        name="ghost", value="no_such_module_for_smc_cli_tests:Thing", group=GROUP
    )
    _register(monkeypatch, extra=(ghost,))
    result = _run("run", "ghost", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 1, result.output
    assert (
        "✗ plugin ghost (no_such_module_for_smc_cli_tests:Thing) could not be "
        "imported: ModuleNotFoundError"
    ) in result.output


@pytest.mark.usefixtures("no_stand")
def test_a_missing_upstream_exits_2_before_the_stand_opens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Reader)
    result = _run("run", "reader", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 2, result.output
    assert (
        "✗ reader reads up, but the run document has no up; smc run --list shows "
        "which plugin provides it"
    ) in result.output
    assert not (tmp_path / "run.json").exists()


@pytest.mark.usefixtures("no_stand")
def test_an_upstream_of_another_profile_exits_2_before_the_stand_opens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The profile of the check is the one the command names, loaded before the stand.
    write_stage(
        tmp_path / "run.json",
        StageRecord(
            plugin="up",
            provides="up",
            attempt="up-1",
            status=StageStatus.done,
            started_utc=datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc),
            profile="another",
            dry_run=False,
            smc_version="0.0.1",
            params={},
            result={"assumed": [], "value": 1, "label": "", "wells": []},
        ),
    )
    _register(monkeypatch, Reader)
    result = _run("run", "reader", "-p", "demo", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 2, result.output
    assert (
        "✗ reader reads up, but up was run on another, this is demo; run up again "
        "on this profile"
    ) in result.output


@pytest.mark.usefixtures("no_stand")
def test_resume_with_nothing_to_resume_exits_2_before_the_stand_opens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Resumable)
    result = _run("run", "resumable", "--run-doc", _doc(tmp_path), "--resume")
    assert result.exit_code == 2, result.output
    assert "✗ cannot resume resumable: nothing to resume" in result.output


# --- a run that opens the stand ----------------------------------------------


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_run_stage_report_writes_the_document_and_prints_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The format is asserted, not positions compared across two opens (FM-47).
    clock = SimpleNamespace(monotonic=lambda: 0.0)
    monkeypatch.setattr(cli, "time", clock)  # all three progress calls within 1 s
    result = _run(
        "run", "stage-report", "-p", "demo", "--run-doc", _doc(tmp_path),
        "--param", "samples=3",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    lines = _lines(result)
    progress = [line for line in lines if line.startswith("stage-report:")]
    assert progress == ["stage-report: 1/3 read", "stage-report: 3/3 read"]
    (done,) = [line for line in lines if line.startswith("✓ ")]
    assert re.fullmatch(
        rf"✓ stage-report done in \d+\.\d s — {re.escape(_doc(tmp_path))}, stage "
        r"stage-report",
        done,
    ), done
    printed = _json_in(result.output)
    assert printed["profile"] == "demo"
    assert len(printed["readings"]) == 3
    assert all(set(r) == {"x_um", "y_um", "z_um"} for r in printed["readings"])
    stored = _stored(tmp_path, "stage-report")
    assert stored.status is StageStatus.done
    assert stored.result == printed
    assert stored.params == {"samples": 3, "interval_ms": 0.0, "include_z": True}


@pytest.mark.demo
@pytest.mark.usefixtures("demo")
def test_a_missing_capability_exits_1_and_keeps_its_brackets(
    tmp_path: Path, noxy_profile: Path
) -> None:
    # FM-36: `[roles.assign]` is advice, and rich would drop it as markup.
    result = _run(
        "run", "stage-report", "-p", str(noxy_profile), "--run-doc", _doc(tmp_path)
    )
    assert result.exit_code == 1, result.output
    assert (
        "✗ stage-report needs XYStage, which this stand does not have: XYStage "
        "needs the 'xy_stage' role, which no loaded device fills"
    ) in result.output
    assert "[roles.assign]" in result.output
    assert not (tmp_path / "run.json").exists()


def test_a_run_prints_its_result_after_the_stand_is_released(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Good)
    result = _run(
        "run", "good", "--run-doc", _doc(tmp_path),
        "--param", "n=7", "--param", "label=µm [x]", "--param", 'wells=["A1","B2"]',
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert fake_stand == [False]
    printed = _json_in(result.output)
    # ensure_ascii=False: the µ and the brackets reach the terminal as they are.
    assert printed == {
        "assumed": [],
        "value": 7,
        "label": "µm [x]",
        "wells": ["A1", "B2"],
    }
    assert "µm [x]" in result.output


def test_a_result_with_a_lone_surrogate_is_printed_with_a_replacement(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-05: an adapter's string; the document keeps it escaped, the screen cannot.
    _register(monkeypatch, Surrogate)
    result = _run("run", "surrogate", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 0, result.output
    assert f'"label": "{chr(0xFFFD)} µm"' in result.output
    assert max((tmp_path / "run.json").read_bytes()) < 0x80


def test_the_check_before_the_stand_uses_the_name_of_the_profile_the_command_names(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stage records the stand's profile name; the readiness check compares it with
    # the name of the profile file, not with the text that was typed after -p.
    profile = tmp_path / "stand.toml"
    profile.write_text('[microscope]\nname = "stand"\n', encoding="utf-8")
    _register(monkeypatch, Up, Reader)
    first = _run("run", "up", "-p", str(profile), "--run-doc", _doc(tmp_path))
    assert first.exit_code == 0, first.output
    assert _stored(tmp_path, "up").profile == "stand"
    second = _run("run", "reader", "-p", str(profile), "--run-doc", _doc(tmp_path))
    assert second.exit_code == 0, second.output
    assert fake_stand == [False, False]


def test_a_dry_run_reads_a_dry_run_and_a_real_run_does_not(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    assert _run("run", "up", "--dry-run", "--run-doc", _doc(tmp_path)).exit_code == 0
    rehearsal = _run("run", "reader", "--dry-run", "--run-doc", _doc(tmp_path))
    assert rehearsal.exit_code == 0, rehearsal.output
    real = _run("run", "reader", "--run-doc", _doc(tmp_path))
    assert real.exit_code == 2, real.output
    assert (
        "✗ reader reads up, but up was a dry run; run it without --dry-run"
    ) in real.output
    assert fake_stand == [True, True]  # the refusal came before a third open


def test_a_dry_run_may_resume_what_a_dry_run_stopped(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Resumable)
    doc = _doc(tmp_path)
    assert _run("run", "resumable", "--dry-run", "--run-doc", doc).exit_code == 130
    resumed = _run("run", "resumable", "--dry-run", "--resume", "--run-doc", doc)
    assert resumed.exit_code == 0, resumed.output
    refused = _run("run", "resumable", "--resume", "--run-doc", doc)
    assert refused.exit_code == 2  # a real run does not resume a rehearsal
    assert fake_stand == [True, True]


def test_the_plugin_version_of_a_run_is_the_distributions(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, installed=True)
    result = _run("run", "stage-report", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 0, result.output
    assert _stored(tmp_path, "stage-report").plugin_version == smc.__version__


@pytest.mark.parametrize(
    ("plugin", "marker"),
    [
        pytest.param(Good, "done in", id="done"),
        pytest.param(SelfStop, "self-stop stopped", id="stopped"),
    ],
)
def test_the_stand_is_released_before_anything_is_said_about_the_outcome(
    stand_events: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plugin: type,
    marker: str,
) -> None:
    # Design §1.10 step 4 and M1 §9: the stand accepts one connection at a time, so
    # it is let go before the result and before any ✓ or ✗ line.
    _register(monkeypatch, plugin)
    _run("run", plugin.name, "--run-doc", _doc(tmp_path))  # type: ignore[attr-defined]
    closed = stand_events.index("stand closed")
    assert not any(marker in line for line in stand_events[:closed])
    assert any(marker in line for line in stand_events[closed + 1 :])


def test_a_stage_two_levels_down_is_warned_about_too(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader, Third)
    for name in ("up", "reader", "third"):
        assert _run("run", name, "--run-doc", _doc(tmp_path)).exit_code == 0
    again = _run("run", "up", "--run-doc", _doc(tmp_path))
    warnings = [line for line in _lines(again) if line.startswith("! ")]
    assert warnings == [
        "! reader read up from an earlier run; run it again",
        "! third read reader from an earlier run; run it again",
    ]


def test_a_plugin_that_does_not_load_does_not_end_the_listing_of_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(monkeypatch, Good, NoSummary)
    result = _run("run", "--list")
    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert lines[0].startswith("good  ")
    assert lines[1] == (
        f"✗ no-summary ({THIS}:NoSummary): summary must be a string, got None"
    )


def test_dry_run_is_passed_to_the_stand_and_recorded(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Good)
    result = _run("run", "good", "--run-doc", _doc(tmp_path), "--dry-run")
    assert result.exit_code == 0, result.output
    assert fake_stand == [True]
    assert _stored(tmp_path, "good").dry_run is True


def test_a_safety_refusal_exits_2_with_its_reason(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Refuses)
    result = _run("run", "refuses", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 2, result.output
    assert "✗ the jog is too long" in result.output
    assert _stored(tmp_path, "refuses").status is StageStatus.failed


def test_a_hardware_error_exits_1(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Boom)
    result = _run("run", "boom", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 1, result.output
    assert "✗ the stage did not answer" in result.output
    stored = _stored(tmp_path, "boom")
    assert (stored.status, stored.error) == (
        StageStatus.failed,
        "HardwareError: the stage did not answer",
    )


def test_a_plugin_run_error_exits_1_with_its_reason(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Unusable)
    result = _run("run", "unusable", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 1, result.output
    assert "✗ unusable: no plate found; run it again" in result.output
    stored = _stored(tmp_path, "unusable")
    assert (stored.status, stored.error) == (
        StageStatus.failed,
        "PluginRunError: unusable: no plate found; run it again",
    )


def test_ctrl_c_exits_130_and_the_record_says_stopped(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, CtrlC)
    result = _run("run", "ctrl-c", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 130, result.output
    # The plugin raised it outside any move: no stop was given up on.
    assert "✗ interrupted — no move was under way" in result.output
    stored = _stored(tmp_path, "ctrl-c")
    assert (stored.status, stored.error) == (
        StageStatus.stopped,
        "KeyboardInterrupt: Ctrl-C",
    )


# --- a stand whose stages refuse stop() (m2 §11.6) ----------------------------


class MovesXY(Base):
    name = provides = "moves-xy"
    requires = (XYStage,)

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        microscope.require(XYStage).move_to_um(10.0, 0.0)
        return EchoResult()


@pytest.fixture
def refusing_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake stand whose XY and Z refuse stop(), as the Ti2-E's (m2 §11.8).

    The XY stage stays busy, and a Ctrl-C lands at the first busy poll of a
    move, inside the Executor's wait: the path that reaches the give-up.
    """

    def open_stand(
        cls: type[Microscope], profile: object = "demo", *, dry_run: bool = False
    ) -> Microscope:
        core = FakeCore.demo_like()
        core.stop_refused = {"XY", "Z"}
        core.busy_devices = {"XY"}
        real, polled = core.deviceBusy, list[str]()

        def busy(label: str) -> bool:
            if not polled:
                polled.append(label)
                raise KeyboardInterrupt
            return real(label)

        core.deviceBusy = busy  # type: ignore[method-assign]
        built = profile if isinstance(profile, Profile) else Profile.demo()
        return Microscope.from_core(cast("CMMCorePlus", core), built, dry_run=dry_run)

    monkeypatch.setattr(Microscope, "open", classmethod(open_stand))


def _declaring_xy(tmp_path: Path) -> str:
    path = tmp_path / "declared.toml"
    path.write_text(
        '[microscope]\nname = "declared"\n\n[safety]\nstop_refused_by = ["xy_stage"]\n',
        encoding="utf-8",
    )
    return str(path)


_XY_DECLARED_LINE = (
    "! the profile declares that this stand refuses stop() on xy_stage ([safety] "
    "stop_refused_by): once sent, a move runs to its target, and Ctrl-C ends smc, "
    "not the move"
)


@pytest.mark.usefixtures("refusing_stand")
def test_ctrl_c_mid_move_on_a_declared_stand_says_the_stage_runs_to_its_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # #206, FM-117: the Ti2-E's stage went the full 20000 µm after a line that
    # said it was stopped (#208).
    _register(monkeypatch, MovesXY)
    profile = _declaring_xy(tmp_path)
    result = _run("run", "moves-xy", "--run-doc", _doc(tmp_path), "-p", profile)
    assert result.exit_code == 130, result.output
    assert _lines(result)[-1] == (
        "✗ interrupted — xy_stage XY refuses a stop on this stand ([safety] "
        "stop_refused_by), so it runs to its target; let it finish or stop it "
        "with the stand's own controls, then read the position (smc stage get, "
        "smc z get) before the next move"
    )
    assert _XY_DECLARED_LINE in _lines(result)
    assert _stored(tmp_path, "moves-xy").status is StageStatus.stopped


def test_smc_run_on_a_declared_stand_prints_the_stop_line(
    stand_events: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Good)
    profile = _declaring_xy(tmp_path)
    result = _run("run", "good", "--run-doc", _doc(tmp_path), "-p", profile)
    assert result.exit_code == 0, result.output
    # Once, while the stand is open, before the plugin runs.
    assert stand_events.count(_XY_DECLARED_LINE) == 1
    assert stand_events.index(_XY_DECLARED_LINE) < stand_events.index("stand closed")


def test_smc_run_on_a_stand_that_declares_nothing_prints_no_stop_line(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Good)
    result = _run("run", "good", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 0, result.output
    assert "! the profile declares" not in result.output


def test_a_plugin_that_stops_itself_exits_130_with_its_name(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, SelfStop)
    result = _run("run", "self-stop", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 130, result.output
    assert "✗ self-stop stopped" in _lines(result)
    assert _stored(tmp_path, "self-stop").status is StageStatus.stopped


def test_resume_continues_a_stopped_stage(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Resumable)
    first = _run("run", "resumable", "--run-doc", _doc(tmp_path))
    assert first.exit_code == 130, first.output
    second = _run("run", "resumable", "--run-doc", _doc(tmp_path), "--resume")
    assert second.exit_code == 0, second.output
    assert _json_in(second.output)["value"] == 2
    assert _stored(tmp_path, "resumable").status is StageStatus.done


def test_a_failed_final_write_still_prints_the_result(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-34: the stage ran; losing the record must not lose what it measured.
    _register(monkeypatch, Good)
    real_replace, calls = files_mod.os.replace, []

    def replace_once(src: Any, dst: Any) -> None:
        calls.append(src)
        if len(calls) > 1:  # the `running` write went through
            raise PermissionError(13, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(files_mod.os, "replace", replace_once)
    monkeypatch.setattr(files_mod.time, "sleep", lambda seconds: None)
    result = _run("run", "good", "--run-doc", _doc(tmp_path), "--param", "n=5")
    assert result.exit_code == 1, result.output
    assert "could not be replaced (Permission denied)" in result.output
    assert "! good ran, but its record could not be written; its result:" in (
        result.output
    )
    assert _json_in(result.output) == {
        "assumed": [],
        "value": 5,
        "label": "x",
        "wells": [],
    }


def test_rerunning_an_upstream_prints_the_stale_warning(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    up = ("run", "up", "--run-doc", _doc(tmp_path))
    reader = ("run", "reader", "--run-doc", _doc(tmp_path))
    first = _run(*up)
    assert "!" not in first.output
    second = _run(*reader)
    assert second.exit_code == 0, second.output
    assert "from an earlier run" not in second.output  # nothing stale yet
    again = _run(*up)
    assert again.exit_code == 0, again.output
    assert _lines(again)[-1] == "! reader read up from an earlier run; run it again"
    # Running the reader again clears it; and a stage that was stale before
    # this run is not warned about twice.
    fresh = _run(*reader)
    assert "from an earlier run" not in fresh.output
    _run(*up)
    third = _run(*up)
    assert third.output.count("! reader read up") == 0


def test_progress_prints_the_first_and_last_lines_and_one_a_second(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ticks = iter([0.0, 0.5, 1.0, 1.5, 1.6, 9.0, 9.1])
    monkeypatch.setattr(
        cli, "time", SimpleNamespace(monotonic=lambda: next(ticks, 99.0))
    )
    _register(monkeypatch, Progressing)
    result = _run("run", "progressing", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 0, result.output
    progress = [line for line in _lines(result) if line.startswith("progressing:")]
    # 1/5 (first), 2/5 and 4/5 skipped (under a second), 3/5 printed (a second
    # after the first), 5/5 (the last, always), then one without a total.
    assert progress == [
        "progressing: 1/5 step",
        "progressing: 3/5 step",
        "progressing: 5/5 step",
        "progressing: 6",
    ]


def test_a_run_adds_no_logging_handler_to_the_smc_logger(
    fake_stand: list[bool], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # FM-46: what a command configures is undone when its context closes.
    logger = logging.getLogger("smc")
    before = (logger.level, list(logger.handlers))
    _register(monkeypatch, Good)
    result = _run("-v", "run", "good", "--run-doc", _doc(tmp_path))
    assert result.exit_code == 0, result.output
    assert (logger.level, list(logger.handlers)) == before
