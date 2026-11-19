"""``smc run NAME --offline``: the plugins that never drive the stand, on a copied run folder (algorithms §4, #566).

The runner pins the width (FM-37) and clears ``SMC_PROFILE`` and ``SMC_PROFILES``,
so an operator's environment never reaches a test (FM-40). Every command here runs
under ``no_stand``: no microscope may be opened, from a profile or from a core, and
no Micro-Manager install is found, so a command that opened a stand and swallowed
the ``CoreError`` would still fail. Test plugins are classes of this module, put in
the registry through ``entry_points``; the five installed offline plugins are
looked at through the registry as installed. The working directory is the
repository's, where ``profiles/`` is: the recorded profile is looked up by name.

The text of a usage error is asserted through ``_plain`` and by phrases that hold
no option name: typer wraps the message and styles ``--option`` inside it (FM-37,
FM-72).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner, Result

import smc.hardware.core as core_mod
import smc.hardware.microscope as microscope_mod
from smc.cli import app
from smc.hardware import Microscope
from smc.hardware.capabilities import XYStage
from smc.plugins import discover
from smc.plugins import registry as registry_mod
from smc.plugins.api import (
    Plugin,
    PluginParams,
    PluginResult,
    RunContext,
    StopRequested,
)
from smc.plugins.registry import GROUP
from smc.run import RunDocument, StageRecord, StageStatus, write_stage

runner = CliRunner(env={"COLUMNS": "200", "SMC_PROFILE": None, "SMC_PROFILES": None})

THIS = __name__
REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PANEL = re.compile(r"[│╭╮╰╯─]")


# --- plugins of this module --------------------------------------------------


class Params(PluginParams):
    n: int = 1


class EchoResult(PluginResult):
    value: int = 0
    offline: bool = False


class Seen(BaseModel):
    """What a checkpoint holds in these tests."""

    done: int = 1


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
        return EchoResult(value=params.n, offline=microscope.is_offline)


class Up(Base):
    name = provides = "up"
    summary = "Produces the slot up."


class Side(Base):
    name = provides = "side"
    summary = "Produces the slot side."


class Reader(Base):
    name = provides = "reader"
    summary = "Reads up; never drives the stand."
    reads = ("up",)
    offline = True


class TwoReader(Base):
    name = provides = "two-reader"
    summary = "Reads up and side; never drives the stand."
    reads = ("up", "side")
    offline = True


class Solo(Base):
    name = provides = "solo"
    summary = "Reads nothing; never drives the stand."
    offline = True


class Driver(Base):
    name = provides = "driver"
    summary = "Needs the XY stage."
    requires = (XYStage,)
    reads = ("up",)


class Truthy(Base):
    name = provides = "truthy"
    summary = "Sets offline to something that is not True."
    reads = ("up",)
    offline = 1  # type: ignore[assignment]


class Boom(Reader):
    name = provides = "boom"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        raise RuntimeError("the detector failed")


class Resumable(Reader):
    """Stops itself after one unit; the attempt resumed from it finishes."""

    name = provides = "resumable"

    def run(self, microscope: Any, params: Params, ctx: RunContext) -> EchoResult:
        if ctx.checkpoint(Seen) is None:
            ctx.save_checkpoint(Seen())
            raise StopRequested
        return EchoResult(value=2, offline=microscope.is_offline)


def _register(monkeypatch: pytest.MonkeyPatch, *classes: type) -> None:
    entries = [
        EntryPoint(name=cls.name, value=f"{THIS}:{cls.__name__}", group=GROUP)  # type: ignore[attr-defined]
        for cls in classes
    ]
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


@pytest.fixture(autouse=True)
def _in_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """The profiles are looked up by name from ``./profiles``: run from the repository folder."""
    monkeypatch.chdir(REPO)


@pytest.fixture(autouse=True)
def no_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command opens a stand, whichever way, and hide the demo adapters.

    ``find_install`` answers ``None`` as on a laptop with no Micro-Manager; the
    failing ``open``, ``from_core`` and ``open_core`` are for a command that
    opened one anyway and swallowed the ``CoreError``.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        pytest.fail("the command opened a stand")

    monkeypatch.setattr(Microscope, "open", classmethod(refuse))
    monkeypatch.setattr(Microscope, "from_core", classmethod(refuse))
    monkeypatch.setattr(core_mod, "open_core", refuse)
    monkeypatch.setattr(microscope_mod, "open_core", refuse)
    monkeypatch.setattr(core_mod, "find_install", lambda: None)


def _stage(
    plugin: str,
    *,
    profile: str = "demo",
    minute: int = 0,
    dry_run: bool = False,
) -> StageRecord:
    started = T0 + timedelta(minutes=minute)
    return StageRecord(
        plugin=plugin,
        provides=plugin,
        attempt=f"{plugin}-1",
        status=StageStatus.done,
        started_utc=started,
        finished_utc=started,
        profile=profile,
        dry_run=dry_run,
        smc_version="0",
        params={},
        result={},
    )


def _seed(tmp_path: Path, *records: StageRecord) -> Path:
    path = tmp_path / "run.json"
    for record in records:
        write_stage(path, record)
    return path


def _run(*args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, list(args), env=env)


def _plain(result: Result) -> str:
    """The output without colour codes, box lines or runs of blanks (FM-72)."""
    return " ".join(_PANEL.sub(" ", _ANSI.sub("", result.output)).split())


def _stored(tmp_path: Path, plugin: str) -> StageRecord:
    return RunDocument.load(tmp_path / "run.json").stages[plugin]


def _text(tmp_path: Path) -> dict[str, Any]:
    """The run document as written, key for key."""
    stages: dict[str, Any] = json.loads(
        (tmp_path / "run.json").read_text(encoding="utf-8")
    )["stages"]
    return stages


# --- which plugins are offline -----------------------------------------------


def _installed() -> dict[str, Plugin[Any, Any]]:
    return {name: entry.plugin for name, entry in discover().plugins.items()}


def test_the_five_plugins_that_never_drive_the_stand_are_the_offline_ones() -> None:
    # The installed registry, not this module's: the entry points of the project.
    assert importlib.metadata.entry_points(group=GROUP)
    plugins = _installed()

    offline = sorted(
        name for name, plugin in plugins.items() if getattr(plugin, "offline", False)
    )

    assert offline == ["cells", "density", "nis-export", "report", "route"]
    # The attribute is a real True, as the command reads it (`is True`).
    assert all(plugins[name].offline is True for name in offline)  # type: ignore[attr-defined]


def test_an_offline_plugin_requires_no_capability() -> None:
    # A plugin that sets `offline` runs on a facade that has no capability: one
    # that required any would fail on every offline run (api.Plugin's docstring).
    for name, plugin in _installed().items():
        if getattr(plugin, "offline", False):
            assert plugin.requires == (), name


# --- a run ---------------------------------------------------------------------


def test_offline_runs_on_the_recorded_profile_and_records_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No --profile, no SMC_PROFILE: the profile is the one the document recorded,
    # which is not the command's default, so a run on the default would be refused.
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up", profile="nikon-ti2"))

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 0, result.output
    stored = _stored(tmp_path, "reader")
    assert stored.status is StageStatus.done
    assert stored.profile == "nikon-ti2"
    assert stored.offline is True
    assert stored.dry_run is False
    # The plugin ran on Microscope.offline: it says so in its result.
    assert stored.result is not None
    assert stored.result["offline"] is True
    text = _plain(result)
    assert (
        "offline: no stand is opened; the profile, nikon-ti2, is the run document's"
        in text
    )
    assert "reader done in" in text
    # nikon-ti2 declares that its stages refuse stop(): the warning is for a
    # stand that is open, and none is.
    assert "refuses stop()" not in text


def test_an_offline_run_leaves_the_stages_of_the_stand_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stage made on the stand keeps no `offline` key in the file, as before
    # the field existed (FM-107); only the offline run's own stage has one.
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up"))
    before = _text(tmp_path)["up"]

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 0, result.output
    stages = _text(tmp_path)
    assert stages["up"] == before
    assert "offline" not in stages["up"]
    assert stages["reader"]["offline"] is True


def test_offline_refuses_a_plugin_that_drives_the_stand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader, Solo, Driver, TwoReader)
    doc = _seed(tmp_path, _stage("up"))

    result = _run("run", "driver", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    text = _plain(result)
    assert "driver drives the stand" in text
    # The offline ones, sorted, from the registry.
    assert "reader, solo, two-reader" in text
    assert "driver" not in _stored_names(tmp_path)


def test_offline_refuses_a_plugin_whose_attribute_is_not_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `offline = 1` is not the opt-in: a plugin says True, or it drives the stand.
    _register(monkeypatch, Up, Truthy, Reader)
    doc = _seed(tmp_path, _stage("up"))

    result = _run("run", "truthy", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    text = _plain(result)
    assert "truthy drives the stand" in text
    assert "runs: reader" in text


def test_offline_with_no_offline_plugin_says_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Driver)
    doc = _seed(tmp_path, _stage("up"))

    result = _run("run", "driver", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    assert "runs: none" in _plain(result)


def _stored_names(tmp_path: Path) -> list[str]:
    return sorted(RunDocument.load(tmp_path / "run.json").stages)


def test_offline_refuses_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up"))

    result = _run("run", "reader", "--offline", "--dry-run", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    text = _plain(result)
    assert "cannot be combined" in text
    assert "an offline run opens no stand, so there is nothing to rehearse" in text
    assert _stored_names(tmp_path) == ["up"]


# --- the profile ---------------------------------------------------------------


@pytest.mark.parametrize("how", ["option", "environment"])
def test_offline_refuses_a_profile_that_differs_from_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, how: str
) -> None:
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up", profile="demo"))
    args = ["run", "reader", "--offline", "--run-doc", str(doc)]
    env = None
    if how == "option":
        args += ["-p", "nikon-ti2"]
    else:
        env = {"SMC_PROFILE": "nikon-ti2"}

    result = _run(*args, env=env)

    assert result.exit_code == 2, result.output
    text = _plain(result)
    # Both names, and what an offline run does instead.
    assert "the run document was made on profile 'demo', this is 'nikon-ti2'" in text
    assert "an offline run uses the profile the document recorded" in text
    assert _stored_names(tmp_path) == ["up"]


@pytest.mark.parametrize("how", ["option", "environment", "path"])
def test_offline_accepts_the_recorded_profile_when_it_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, how: str
) -> None:
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up", profile="nikon-ti2"))
    args = ["run", "reader", "--offline", "--run-doc", str(doc)]
    env = None
    if how == "option":
        args += ["-p", "nikon-ti2"]
    elif how == "environment":
        env = {"SMC_PROFILE": "nikon-ti2"}
    else:
        # A path to the profile's file is accepted: its [microscope] name is the one.
        args += ["-p", str(REPO / "profiles" / "nikon-ti2.toml")]

    result = _run(*args, env=env)

    assert result.exit_code == 0, result.output
    assert _stored(tmp_path, "reader").profile == "nikon-ti2"


def test_offline_with_a_given_profile_that_does_not_exist_is_the_usual_profile_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up"))

    result = _run(
        "run", "reader", "--offline", "--run-doc", str(doc), "-p", "no-such-profile"
    )

    assert result.exit_code == 1, result.output
    assert "no-such-profile" in _plain(result)
    assert _stored_names(tmp_path) == ["up"]


def test_offline_refuses_a_recorded_profile_that_is_a_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A name that Profile.load would read as a file (a separator, a .toml): the
    # record is a value from a file, looked up by name only (FM-125).
    _register(monkeypatch, Up, Reader)
    # os.sep is the platform's own separator: "\\" is one on Windows only, where
    # Profile.load reads it as a path too.
    for recorded in ("profiles/demo.toml", "demo.toml", f"..{os.sep}demo"):
        doc = _seed(tmp_path, _stage("up", profile=recorded))

        result = _run("run", "reader", "--offline", "--run-doc", str(doc))

        assert result.exit_code == 2, (recorded, result.output)
        assert (
            f"the run document names the profile {recorded!r}, which is not a "
            f"profile name"
        ) in _plain(result)
        assert _stored_names(tmp_path) == ["up"]


def test_offline_says_which_profile_the_document_recorded_when_it_is_not_on_the_search_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up", profile="not-a-profile"))

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    text = _plain(result)
    assert "the run document was made on profile 'not-a-profile': " in text
    assert "not-a-profile" in text.split("the run document was made on profile")[1]
    assert _stored_names(tmp_path) == ["up"]


def test_offline_refuses_providers_of_two_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Side, TwoReader)
    doc = _seed(
        tmp_path, _stage("up", profile="demo"), _stage("side", profile="nikon-ti2")
    )

    result = _run("run", "two-reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    assert (
        "the run document's stages were made on different profiles (demo, "
        "nikon-ti2); an offline run needs one"
    ) in _plain(result)


def test_offline_takes_the_profile_of_the_stages_the_plugin_reads_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stage of another profile that the plugin does not read decides nothing.
    _register(monkeypatch, Up, Side, Reader)
    doc = _seed(
        tmp_path, _stage("up", profile="demo"), _stage("side", profile="nikon-ti2")
    )

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 0, result.output
    assert _stored(tmp_path, "reader").profile == "demo"


# --- the run document ----------------------------------------------------------


def test_offline_refuses_a_run_document_with_nothing_upstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Side, Reader, Solo)
    doc = _seed(tmp_path, _stage("side"))

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    assert (
        "nothing upstream: an offline run reads a run document made at a stand"
    ) in _plain(result)
    # A plugin that reads nothing has nothing upstream, whatever the document.
    solo = _run("run", "solo", "--offline", "--run-doc", str(doc))
    assert solo.exit_code == 2, solo.output
    assert "nothing upstream" in _plain(solo)
    assert _stored_names(tmp_path) == ["side"]


def test_offline_refuses_a_run_document_that_does_not_exist_and_does_not_create_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    missing = tmp_path / "copied.json"

    result = _run("run", "reader", "--offline", "--run-doc", str(missing))

    assert result.exit_code == 2, result.output
    text = _plain(result)
    assert "does not exist: an offline run reads a run document made at a stand" in text
    assert not missing.exists()


def test_offline_needs_a_run_document_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(monkeypatch, Up, Reader)

    result = _run("run", "reader", "--offline")

    assert result.exit_code == 2, result.output
    assert "a run needs" in _plain(result)


def test_offline_leaves_an_unreadable_run_document_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader)
    doc = tmp_path / "run.json"
    doc.write_text("{ this is not json", encoding="utf-8")

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    # The usual run-document error, not a usage error: nothing the command line says.
    assert result.exit_code == 1, result.output
    assert doc.read_text(encoding="utf-8") == "{ this is not json"


def test_an_upstream_that_was_a_dry_run_is_refused_offline_as_on_a_stand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # check_ready runs on the document with dry_run false: the upstream rule is
    # the stand's, and a dry run's result is not a measurement.
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up", dry_run=True))

    result = _run("run", "reader", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 2, result.output
    assert "was a dry run" in _plain(result)
    assert _stored_names(tmp_path) == ["up"]


# --- parameters, preset, resume ------------------------------------------------


def test_offline_resume_and_preset_work_as_on_a_stand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader, Resumable)
    doc = _seed(tmp_path, _stage("up"))
    preset = tmp_path / "mine.toml"
    preset.write_text('[params]\nreader = ["n=7"]\n', encoding="utf-8")

    # --param wins over the preset, and a preset's value reaches the run.
    ran = _run(
        "run", "reader", "--offline", "--run-doc", str(doc), "--preset", str(preset)
    )
    assert ran.exit_code == 0, ran.output
    stored = _stored(tmp_path, "reader")
    assert stored.params == {"n": 7}
    assert stored.preset == "mine"
    assert stored.offline is True
    over = _run(
        "run", "reader", "--offline", "--run-doc", str(doc), "--preset", str(preset),
        "--param", "n=9",
    )  # fmt: skip
    assert over.exit_code == 0, over.output
    assert _stored(tmp_path, "reader").params == {"n": 9}

    # A finished stage cannot be resumed, offline either.
    again = _run("run", "reader", "--offline", "--run-doc", str(doc), "--resume")
    assert again.exit_code == 2, again.output
    assert "cannot resume reader: reader finished" in _plain(again)

    # A stage that stopped itself is resumed offline, from its checkpoint.
    first = _run("run", "resumable", "--offline", "--run-doc", str(doc))
    assert first.exit_code == 130, first.output
    assert _stored(tmp_path, "resumable").status is StageStatus.stopped
    resumed = _run("run", "resumable", "--offline", "--run-doc", str(doc), "--resume")
    assert resumed.exit_code == 0, resumed.output
    done = _stored(tmp_path, "resumable")
    assert done.status is StageStatus.done
    assert done.offline is True
    assert done.result is not None
    assert done.result["value"] == 2


def test_offline_a_plugin_that_fails_is_recorded_failed_and_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Boom)
    doc = _seed(tmp_path, _stage("up"))

    result = _run("run", "boom", "--offline", "--run-doc", str(doc))

    assert result.exit_code == 1, result.output
    stored = _stored(tmp_path, "boom")
    assert stored.status is StageStatus.failed
    assert stored.offline is True
    assert stored.error is not None
    assert "the detector failed" in stored.error


def test_list_and_show_params_ignore_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _register(monkeypatch, Up, Reader, Driver)

    listed = _run("run", "--list", "--offline")
    shown = _run("run", "driver", "--show-params", "--offline")

    # Neither refuses the plugin that drives the stand, reads a run document or
    # needs one named.
    assert listed.exit_code == 0, listed.output
    assert "driver" in listed.output
    assert shown.exit_code == 0, shown.output
    assert shown.output.split()[0] == "n"


def test_without_offline_a_run_is_unchanged_and_says_nothing_of_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stand's path: `Microscope.open` is the failing one of no_stand, so the
    # command reaches it and no earlier check moved.
    _register(monkeypatch, Up, Reader)
    doc = _seed(tmp_path, _stage("up"))

    with pytest.raises(pytest.fail.Exception, match="the command opened a stand"):
        _run("run", "reader", "--run-doc", str(doc))
