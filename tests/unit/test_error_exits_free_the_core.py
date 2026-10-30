"""An error exit must free the core by reference counting alone (#148, FM-92).

The cyclic collector runs at an arbitrary moment, and at the exit of a command
during interpreter finalization, where a thread still inside MMCore aborts the
process (#142). #142 made the success paths free the stand; the error paths
kept it alive, because an exception that is kept where its own traceback
reaches it is a cycle: exception, traceback, frame, local, exception.

Every test here runs under ``no_cyclic_gc`` and watches the facade and the core
with weak references. A new error path of a stop, a safe call or a CLI exit
goes in here. Hazards, measured on #148:

* Pytest keeps every ``LogRecord`` of a test, and the arguments of the
  record of a failed stop hold the error, whose traceback holds the fake: a
  retention by the test's handler, not a cycle of the code, so logging is
  disabled here (``_no_kept_log_records``).
* ``FakeCore.failing`` keeps the instance it raised, and its traceback holds
  the fake's frames: clear it before the drop.
* A hook is an instance attribute of the fake, removed with ``del``;
  ``monkeypatch.setattr`` on the core keeps the original bound method, and so
  the core, until teardown.
* ``CliRunner`` adds a cycle of its own (through ``SystemExit``), and
  ``main()`` collects after the command: the CLI is driven through
  ``cli.app(...)`` inside ``try/except SystemExit``.
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import types
import weakref
from collections.abc import Callable, Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
import tifffile
import typer
from pymmcore_plus import CMMCorePlus

import smc.cli as cli
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, XYStage, ZStage
from smc.hardware.errors import HardwareError, SafetyRefusedError, StopRefusedError
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.hardware.safety import Executor, Motion
from smc.plugins import registry as registry_mod
from smc.plugins.api import PluginParams, PluginResult, RunContext, StopRequested
from smc.plugins.registry import GROUP
from smc.testing import FakeCore, PlateSample, SampleCamera
from smc.ui.controller import PageServerError

_FACADE_AND_CORE_FREED = {"facade": False, "core": False}


@pytest.fixture(autouse=True)
def _no_kept_log_records() -> Iterator[None]:
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _watch(microscope: Microscope, core: FakeCore) -> dict[str, weakref.ref[Any]]:
    return {"facade": weakref.ref(microscope), "core": weakref.ref(core)}


def _alive(watched: dict[str, weakref.ref[Any]]) -> dict[str, bool]:
    return {name: ref() is not None for name, ref in watched.items()}


def _stand(
    *,
    dry_run: bool = False,
    declares: tuple[Role, ...] = (),
    engage_drives_z: bool = False,
    device_timeout_ms: int | None = None,
) -> tuple[FakeCore, Microscope]:
    """A fake demo stand; ``declares`` goes into ``[safety] stop_refused_by``.

    ``engage_drives_z`` sets ``[autofocus] engage_drives_z_to_offset`` (m2 §12),
    ``device_timeout_ms`` ``[micromanager] device_timeout_ms``: how long a move
    waits before it gives up.
    """
    core = FakeCore.demo_like()
    profile = Profile.demo()
    if declares:
        safety = profile.safety.model_copy(update={"stop_refused_by": declares})
        profile = profile.model_copy(update={"safety": safety})
    if device_timeout_ms is not None:
        micromanager = profile.micromanager.model_copy(
            update={"device_timeout_ms": device_timeout_ms}
        )
        profile = profile.model_copy(update={"micromanager": micromanager})
    if engage_drives_z:
        autofocus = profile.autofocus.model_copy(
            update={"engage_drives_z_to_offset": True}
        )
        profile = profile.model_copy(update={"autofocus": autofocus})
    return core, Microscope.from_core(
        cast("CMMCorePlus", core), profile, dry_run=dry_run
    )


def _outcome(action: Callable[[], object]) -> tuple[str, str]:
    """The type name and the message of what ``action`` raised; ``("", "")`` if nothing.

    The exception is not kept: it would keep every frame it passed through,
    the test's own included.
    """
    try:
        action()
    except BaseException as exc:  # a second Ctrl-C is one too
        return type(exc).__name__, str(exc)
    return "", ""


# --- main() collects what a command left in a cycle -------------------------


class _Cycle:
    """An object that refers to itself: only the cyclic collector frees it."""

    def __init__(self) -> None:
        self.itself = self


def test_main_collects_what_a_command_left_in_a_cycle_before_it_exits(
    no_cyclic_gc: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #142: the cyclic collector must not be left to interpreter finalization.
    watched: list[weakref.ref[_Cycle]] = []

    def fake_app() -> None:
        cycle = _Cycle()
        watched.append(weakref.ref(cycle))
        raise SystemExit(3)

    monkeypatch.setattr(cli, "app", fake_app)
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 3
    assert [ref() is not None for ref in watched] == [False]


def test_main_lets_an_exception_that_escapes_app_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `--debug`: the traceback is what the operator asked for, and the exit
    # code is Python's. Only a SystemExit is held back for the collection.
    def fake_app() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "app", fake_app)
    with pytest.raises(RuntimeError, match=r"^boom$"):
        cli.main()


# --- a CLI command that ends in an error frees its core ---------------------


class _FaultyXY(FakeCore):
    """A stand whose XY position read fails with a fresh error every time.

    Not ``failing``: that keeps the one instance it raised, and its traceback
    holds the fake's frames, a cycle through the core of the test's own making.
    """

    def getXYPosition(self, label: str | None = None) -> list[float]:  # noqa: N802
        raise RuntimeError("device fault")


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    core_type: type[FakeCore],
    args: list[str],
    stand_profile: Profile | None = None,
) -> tuple[int | str | None, dict[str, weakref.ref[Any]]]:
    """Run ``smc <args>`` on a fake demo stand; the exit code and what to watch.

    Through ``cli.app``, not ``main()`` (its collection would hide the cycle)
    and not ``CliRunner`` (it adds a cycle of its own). ``stand_profile`` is the
    profile the fake stand is built with (the demo's by default).
    """
    watched: dict[str, weakref.ref[Any]] = {}

    def open_stand(
        cls: type[Microscope], profile: str = "demo", *, dry_run: bool = False
    ) -> Microscope:
        core = core_type.demo_like()
        microscope = Microscope.from_core(
            cast("CMMCorePlus", core),
            stand_profile if stand_profile is not None else Profile.demo(),
            dry_run=dry_run,
        )
        watched.update(_watch(microscope, core))
        return microscope

    monkeypatch.setattr(Microscope, "open", classmethod(open_stand))
    code: int | str | None = None
    try:
        cli.app(args, prog_name="smc")
    except SystemExit as exc:
        code = exc.code
    return code, watched


@pytest.mark.parametrize(
    ("args", "core_type", "code"),
    [
        pytest.param(["stage", "jog", "100000", "0"], FakeCore, 2, id="xy-jog-refused"),
        # The demo profile's Z jog limit is 100 µm.
        pytest.param(["z", "jog", "1e6"], FakeCore, 2, id="z-jog-refused"),
        pytest.param(["stage", "get"], _FaultyXY, 1, id="device-fault"),
    ],
)
def test_a_cli_command_that_ends_in_an_error_frees_its_core(
    no_cyclic_gc: None,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    core_type: type[FakeCore],
    code: int,
) -> None:
    # `_Reported` was a generator: one that converts an exception keeps the
    # command's frames, and the stand, in a cycle on Python 3.11+ (#148).
    exit_code, watched = _run_cli(monkeypatch, core_type, args)
    assert exit_code == code
    assert _alive(watched) == _FACADE_AND_CORE_FREED


def test_snap_whose_file_cannot_be_written_frees_its_core(
    no_cyclic_gc: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The frame was taken, so `snap` keeps the write error to print it after
    # the stand is released: that local is on its own traceback (FM-34, #148).
    def denied(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(tifffile, "imwrite", denied)
    out = tmp_path / "frame.tif"
    exit_code, watched = _run_cli(monkeypatch, FakeCore, ["snap", "--out", str(out)])
    assert exit_code == 1
    assert _alive(watched) == _FACADE_AND_CORE_FREED


def test_a_cli_command_that_succeeds_frees_its_core(
    no_cyclic_gc: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pins that the mechanism of the tests above frees a core at all.
    exit_code, watched = _run_cli(monkeypatch, FakeCore, ["stage", "get"])
    assert exit_code == 0
    assert _alive(watched) == _FACADE_AND_CORE_FREED


# --- `smc run`: an exit while the stand is open frees it ---------------------
# The runner re-raises what the plugin raised, so the error's traceback holds the
# runner's frame, which holds the stand: nothing may keep the error where that
# frame reaches it (FM-98), and the context holds no reference to the stand (FM-92).


class _RunParams(PluginParams):
    pass


class _RunResult(PluginResult):
    pass


class RunOk:
    name = provides = "run-ok"
    summary = "Succeeds."
    Params = _RunParams
    Result = _RunResult
    requires: tuple[type, ...] = ()
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> _RunResult:
        ctx.progress(1, 1, "done")
        return _RunResult()


class RunRefused(RunOk):
    name = provides = "run-refused"

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> _RunResult:
        raise SafetyRefusedError("the jog is too long")


class RunFault(RunOk):
    name = provides = "run-fault"

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> _RunResult:
        raise HardwareError("the stage did not answer")


class RunCtrlC(RunOk):
    name = provides = "run-ctrl-c"

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> _RunResult:
        raise KeyboardInterrupt


class RunStops(RunOk):
    name = provides = "run-stops"

    def run(self, microscope: Any, params: Any, ctx: RunContext) -> _RunResult:
        raise StopRequested


class RunNeedsXY(RunOk):
    name = provides = "run-needs-xy"
    requires = (XYStage,)


def _register(monkeypatch: pytest.MonkeyPatch, *classes: type[RunOk]) -> None:
    entries = [
        EntryPoint(name=c.name, value=f"{__name__}:{c.__name__}", group=GROUP)
        for c in classes
    ]
    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: list(entries))


@pytest.mark.parametrize(
    ("plugin", "code"),
    [
        pytest.param(RunOk, 0, id="done"),
        pytest.param(RunRefused, 2, id="safety-refused"),
        pytest.param(RunFault, 1, id="hardware-error"),
        pytest.param(RunCtrlC, 130, id="ctrl-c"),
        pytest.param(RunStops, 130, id="plugin-stop"),
    ],
)
def test_a_smc_run_that_ends_frees_its_core(
    no_cyclic_gc: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plugin: type[RunOk],
    code: int,
) -> None:
    _register(monkeypatch, plugin)
    args = ["run", plugin.name, "--run-doc", str(tmp_path / "run.json")]
    exit_code, watched = _run_cli(monkeypatch, FakeCore, args)
    assert exit_code == code
    assert _alive(watched) == _FACADE_AND_CORE_FREED


def test_a_smc_run_that_lacks_a_capability_frees_its_core(
    no_cyclic_gc: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The runner refuses before it writes anything: a RequirementsError raised
    # with the stand open, exit 1.
    profile_file = tmp_path / "noxy.toml"
    profile_file.write_text(
        '[microscope]\nname = "noxy"\n\n[roles.exclude]\nxy_stage = ["XY"]\n',
        encoding="utf-8",
    )
    _register(monkeypatch, RunNeedsXY)
    args = ["run", RunNeedsXY.name, "--run-doc", str(tmp_path / "run.json")]
    exit_code, watched = _run_cli(
        monkeypatch, FakeCore, args, stand_profile=Profile.load(profile_file)
    )
    assert exit_code == 1
    assert _alive(watched) == _FACADE_AND_CORE_FREED
    assert not (tmp_path / "run.json").exists()


def _context() -> typer.Context:
    return typer.Context(typer.main.get_command(cli.app))


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: typer.Exit(code=3), id="exit"),
        pytest.param(typer.Abort, id="abort"),
        pytest.param(lambda: typer.BadParameter("x"), id="bad-parameter"),
        pytest.param(lambda: SystemExit(4), id="system-exit"),
        pytest.param(GeneratorExit, id="generator-exit"),
    ],
)
def test_reported_lets_what_is_not_a_failure_of_the_command_through(
    make: Callable[[], BaseException], capsys: pytest.CaptureFixture[str]
) -> None:
    # Click's own flow (an exit, an abort, a bad parameter) and what is not an
    # error of the command's own (a SystemExit) keep their meaning.
    error = make()
    with pytest.raises(type(error)) as raised, cli._Reported(_context()):
        raise error
    assert raised.value is error
    shown = capsys.readouterr()
    assert (shown.out, shown.err) == ("", "")


# --- a failed stop or switch-off frees the core -----------------------------

_XY_STOP_FAILED = (
    "xy_stage: stop('XY') failed (device fault); check the device: power, cable, "
    "another program holding it"
)


def _xy_stop(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    core.failing["XY"] = RuntimeError("device fault")
    return _outcome(microscope.require(XYStage).stop)


def _autofocus_disengage(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    autofocus = microscope.require(Autofocus)
    core.enableContinuousFocus(True)
    core.failing["Autofocus"] = RuntimeError("device fault")
    return _outcome(autofocus.disengage)


def _emergency_stop(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    core.failing["XY"] = RuntimeError("device fault")
    return _outcome(microscope.stop)


def _z_stop_before_the_move(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    """A Z stop that lands inside the move's switch-off of PFS: the move is cancelled."""
    core.autofocus_lockable = True
    core.enableContinuousFocus(True)
    z = microscope.require(ZStage)
    original = core.enableContinuousFocus

    def switch(enable: bool) -> None:
        # FM-94: nothing is asserted here, the layer swallows what a hook raises.
        original(enable)
        if not enable:
            z.stop()

    core.enableContinuousFocus = switch  # type: ignore[method-assign]
    try:
        return _outcome(lambda: z.move_to_um(3.0))
    finally:
        # Removed, not restored: assigning the bound method back is a cycle.
        del core.enableContinuousFocus


def _z_stop_once_locked(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    """The real PFS link: a Z stop once the relock has locked, whose switch-off fails (#163)."""
    core.autofocus_lockable = True
    core.enableContinuousFocus(True)
    z = microscope.require(ZStage)
    original, stopped = core.enableContinuousFocus, list[bool]()

    def switch(enable: bool) -> None:
        if not enable and stopped:
            raise RuntimeError("device fault")  # fresh: not `failing` (FM-98)
        original(enable)
        if enable and not stopped:
            stopped.append(True)
            _outcome(z.stop)  # the stopper's own failure is not the subject

    core.enableContinuousFocus = switch  # type: ignore[method-assign]
    try:
        return _outcome(lambda: z.move_to_um(3.0))
    finally:
        del core.enableContinuousFocus


def _z_stop_that_fails_during_the_relock(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str]:
    """The real PFS link: a Z stop during the relock whose own ``stop('Z')`` fails (#163).

    `_stop_linked` kept the error `_send_stop` returned; its traceback reaches
    that frame through `_send_safe`'s `f_back`.
    """
    core.autofocus_lockable = True
    core.enableContinuousFocus(True)
    z = microscope.require(ZStage)
    switch, stop, stopped = core.enableContinuousFocus, core.stop, list[bool]()

    def stop_once_on(enable: bool) -> None:
        switch(enable)
        if enable and not stopped:
            stopped.append(True)
            _outcome(z.stop)  # the stopper's own failure is not the subject

    def z_stop_fails(label: str) -> None:
        if label == "Z":
            raise RuntimeError("device fault")  # fresh: not `failing` (FM-98)
        stop(label)

    core.enableContinuousFocus = stop_once_on  # type: ignore[method-assign]
    core.stop = z_stop_fails  # type: ignore[method-assign]
    try:
        return _outcome(lambda: z.move_to_um(3.0))
    finally:
        del core.enableContinuousFocus, core.stop


def _lock_wait_given_up(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    """A lock wait given up whose switch-off fails, then ``stop()`` and ``resume()`` (#163).

    Until then the wait stays registered, and its motion keeps the core
    through the Executor: by decision, not a cycle of an exception (FM-92).
    """
    core.autofocus_lockable = False
    autofocus = microscope.require(Autofocus)
    original = core.enableContinuousFocus

    def switch(enable: bool) -> None:
        if not enable:
            raise RuntimeError("device fault")
        original(enable)

    core.enableContinuousFocus = switch  # type: ignore[method-assign]
    try:
        outcome = _outcome(lambda: autofocus.engage(timeout_s=0.1))
    finally:
        del core.enableContinuousFocus
    microscope.stop()
    microscope.resume()
    return outcome


_SWITCH_OFF_FAILED = (
    "autofocus: enableContinuousFocus(False) failed (device fault); check the "
    "device: power, cable, another program holding it"
)

# --- the engage guard (m2 §12) ------------------------------------------------
# The guard keeps a trip and a failed Z read as text; the error is built in the
# readback, and a failed switch-off goes through the list the action empties.


def _engage_excursion(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    """Run 1 on the demo's limits: locked, then Z driven past the soft limit."""
    core.autofocus_target_um = lambda: 5000.0
    return _outcome(microscope.require(Autofocus).engage)


def _engage_excursion_not_switched_off(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str]:
    """The same jump, whose switch-off fails: registered until ``stop()`` and ``resume()``."""
    core.autofocus_target_um = lambda: 5000.0
    autofocus = microscope.require(Autofocus)
    original = core.enableContinuousFocus

    def switch(enable: bool) -> None:
        if not enable:
            raise RuntimeError("device fault")  # fresh: not `failing` (FM-98)
        original(enable)

    core.enableContinuousFocus = switch  # type: ignore[method-assign]
    try:
        outcome = _outcome(autofocus.engage)
    finally:
        del core.enableContinuousFocus
    microscope.stop()
    microscope.resume()
    return outcome


def _engage_z_unreadable(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    """Z read before the switch-on, then failing once the autofocus is on."""
    autofocus = microscope.require(Autofocus)
    read = core.getPosition

    def position(label: str | None = None) -> float:
        if core.isContinuousFocusEnabled():
            raise RuntimeError("device fault")  # fresh: not `failing` (FM-98)
        return read(label)

    core.getPosition = position  # type: ignore[method-assign]
    try:
        return _outcome(autofocus.engage)
    finally:
        del core.getPosition


def _engage_refused_after_switch_off(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str]:
    """On a declared stand, searching at an offset never seen: switched off, then refused."""
    autofocus = microscope.require(Autofocus)
    core.autofocus_lockable = False
    core.enableContinuousFocus(True)
    return _outcome(autofocus.engage)


_engage_refused_after_switch_off.engage_drives_z = True  # type: ignore[attr-defined]

_ENGAGE_EXCURSION = (
    "autofocus Autofocus: Z went from 0.00 µm to 5000.00 µm while it engaged, "
    "outside the soft limits [-1000, 1000] µm"
)

# --- a stop the profile declares refused (m2 §11) ---------------------------
# `_send_stop` builds a StopRefusedError whose cause is the device's error, and
# that error's traceback reaches `_send_stop`'s frame through `_send_safe`'s
# `f_back` (#163); `_stop_stages` collects the refusals as text.

Scenario = Callable[[FakeCore, Microscope], tuple[str, str]]


def _declaring(*roles: Role) -> Callable[[Scenario], Scenario]:
    """Run the scenario on a stand whose profile declares ``roles`` (read by the test)."""

    def mark(scenario: Scenario) -> Scenario:
        scenario.declares = roles  # type: ignore[attr-defined]
        return scenario

    return mark


@_declaring(Role.xy_stage)
def _declared_xy_stop(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    core.stop_refused = {"XY"}  # a fresh error each time, unlike `failing`
    return _outcome(microscope.require(XYStage).stop)


@_declaring(Role.xy_stage, Role.focus)
def _declared_emergency_stop(core: FakeCore, microscope: Microscope) -> tuple[str, str]:
    core.stop_refused = {"XY", "Z"}
    return _outcome(microscope.stop)


def _timing_out_after(device_timeout_ms: int) -> Callable[[Scenario], Scenario]:
    """Run the scenario on a stand whose moves give up after ``device_timeout_ms``."""

    def mark(scenario: Scenario) -> Scenario:
        scenario.device_timeout_ms = device_timeout_ms  # type: ignore[attr-defined]
        return scenario

    return mark


@_declaring(Role.xy_stage)
@_timing_out_after(50)
def _ctrl_c_inside_the_timeouts_give_up(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str]:
    """A Ctrl-C in the stop that the move's timeout sends: noted, not given up on twice.

    The motion stays registered (it reads busy) until ``resume()``, and keeps
    the core through the Executor by decision, as ``_ctrl_c_whose_stop_is_refused``'s
    (#224).
    """
    core.busy_devices = {"XY"}

    def cut(label: str) -> None:
        raise KeyboardInterrupt("Ctrl-C in the stop")

    core.stop = cut  # type: ignore[method-assign]
    try:
        outcome = _outcome(lambda: microscope.require(XYStage).move_to_um(10.0, 0.0))
    finally:
        del core.stop
    microscope.resume()
    return outcome


@_declaring(Role.focus)
def _declared_z_stop_whose_link_refuses_too(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str]:
    """A Z stop whose link refuses as well: one refusal, built by ``_merged_refusal``.

    The link's error and the device's own both reach ``_stop_linked``'s frame
    through their tracebacks; the merged refusal holds neither (#224, FM-98).
    """
    core.stop_refused = {"Z"}

    def link() -> None:
        raise StopRefusedError(("autofocus_offset PFSOffset",))

    with microscope._executor.linked_stop("Z", link):
        return _outcome(microscope.require(ZStage).stop)


@_declaring(Role.xy_stage)
def _ctrl_c_whose_stop_is_refused(
    core: FakeCore, microscope: Microscope
) -> tuple[str, str]:
    """A Ctrl-C during an XY move that runs on: the give-up's note rides on it.

    The motion stays registered (it reads busy) until ``resume()``, and keeps
    the core through the Executor by decision, as ``_lock_wait_given_up``'s.
    """
    core.stop_refused, core.busy_devices = {"XY"}, {"XY"}
    real, polled = core.deviceBusy, list[str]()

    def busy(label: str) -> bool:
        if not polled:
            polled.append(label)
            raise KeyboardInterrupt
        return real(label)

    core.deviceBusy = busy  # type: ignore[method-assign]
    try:
        outcome = _outcome(lambda: microscope.require(XYStage).move_to_um(10.0, 0.0))
    finally:
        del core.deviceBusy
    microscope.resume()
    return outcome


_DECLARED_XY = (
    "`xy_stage XY` refused the stop, as the profile declares ([safety] "
    "stop_refused_by): a move under way runs to its target; stop it with the "
    "stand's own controls if it must not finish"
)
_DECLARED_Z_AND_ITS_LINK = (
    "`autofocus_offset PFSOffset` and `z Z` refused the stop, as the profile "
    "declares ([safety] stop_refused_by): a move under way runs to its target; "
    "stop it with the stand's own controls if it must not finish"
)
_DECLARED_EMERGENCY_STOP = (
    "the emergency stop halted the microscope and switched autofocus Autofocus "
    "off, but xy_stage XY and z Z refused the stop, as the profile declares "
    "([safety] stop_refused_by): a move under way runs to its target. The "
    "microscope stays halted: stop the stage with the stand's own controls if it "
    "must not finish its move, then call `resume()`"
)


@pytest.mark.parametrize(
    ("scenario", "kind", "message"),
    [
        pytest.param(_xy_stop, "HardwareError", _XY_STOP_FAILED, id="xy-stop"),
        pytest.param(
            _autofocus_disengage,
            "HardwareError",
            "autofocus: enableContinuousFocus(False) failed (device fault); check "
            "the device: power, cable, another program holding it",
            id="autofocus-disengage",
        ),
        pytest.param(
            _emergency_stop,
            "HardwareError",
            "the emergency stop failed for XYStage 'XY' "
            f"(HardwareError({_XY_STOP_FAILED!r})). The microscope stays halted: "
            "stop the stage at the stand if it still moves, then call `resume()`",
            id="emergency-stop",
        ),
        pytest.param(
            _z_stop_before_the_move,
            "MotionStoppedError",
            "`z Z`: it was stopped before it started; read the position before "
            "moving on",
            id="z-stop-before-the-move",
        ),
        pytest.param(
            _z_stop_once_locked,
            "MotionStoppedError",
            "`z Z`: it was stopped during z: move_to 3.0 µm, but autofocus Autofocus "
            f"could not be switched off ({_SWITCH_OFF_FAILED}); it may still move Z: "
            "stop() and check the stand, then resume(); read the position before "
            "moving on",
            id="z-stop-once-locked",
        ),
        pytest.param(
            _z_stop_that_fails_during_the_relock,
            "MotionStoppedError",
            "`z Z`: it was stopped during z: move_to 3.0 µm, so autofocus Autofocus "
            "was left off; read the position before moving on",
            id="z-stop-fails",
        ),
        pytest.param(
            _lock_wait_given_up,
            "DeviceTimeoutError",
            "autofocus Autofocus: no lock within 0.1 s, and it could not be switched "
            f"off (the switch-off failed: {_SWITCH_OFF_FAILED}); it may still move Z: "
            "stop() and check the stand, then resume()",
            id="lock-wait-given-up",
        ),
        pytest.param(
            _declared_xy_stop, "StopRefusedError", _DECLARED_XY, id="declared-xy-stop"
        ),
        pytest.param(
            _declared_emergency_stop,
            "StopRefusedError",
            _DECLARED_EMERGENCY_STOP,
            id="declared-emergency-stop",
        ),
        pytest.param(
            _ctrl_c_inside_the_timeouts_give_up,
            "KeyboardInterrupt",
            "Ctrl-C in the stop",
            id="ctrl-c-inside-the-timeouts-give-up",
        ),
        pytest.param(
            _declared_z_stop_whose_link_refuses_too,
            "StopRefusedError",
            _DECLARED_Z_AND_ITS_LINK,
            id="declared-z-stop-whose-link-refuses-too",
        ),
        pytest.param(
            _ctrl_c_whose_stop_is_refused,
            "KeyboardInterrupt",
            "",
            id="ctrl-c-whose-stop-is-refused",
        ),
        pytest.param(
            _engage_excursion,
            "AutofocusExcursionError",
            f"{_ENGAGE_EXCURSION}; it was switched off. The guard reads Z every 10 "
            "ms: it stops Autofocus once a jump is read, it does not prevent one. "
            "Check the sample and the objective",
            id="engage-excursion",
        ),
        pytest.param(
            _engage_excursion_not_switched_off,
            "_LostLockError",
            f"{_ENGAGE_EXCURSION}, and it could not be switched off (the switch-off "
            f"failed: {_SWITCH_OFF_FAILED}); it may still move Z: stop() and check "
            "the stand, then resume()",
            id="engage-excursion-not-switched-off",
        ),
        pytest.param(
            _engage_z_unreadable,
            "HardwareError",
            "autofocus Autofocus: Z could not be read while it engaged (focus: "
            "getPosition('Z') failed (device fault); check the device: power, "
            "cable, another program holding it); it was switched off, since unread "
            "it could drive Z past any bound (m2 §12)",
            id="engage-z-unreadable",
        ),
        pytest.param(
            _engage_refused_after_switch_off,
            "_EngageRefusedError",
            "autofocus Autofocus: not engaged: on this stand engaging drives Z to "
            "the plane the offset names ([autofocus] engage_drives_z_to_offset, m2 "
            "§12), and offset 0 has not been seen locked with objective "
            "'Nikon 10X S Fluor' since the stand was opened (seen: none); it was on "
            "and searching, so it was switched off. Focus on the sample, switch "
            "Autofocus on with the stand's own controls and set the offset until "
            "the sample is sharp; smc then reads the lock",
            id="engage-refused-after-switch-off",
        ),
    ],
)
def test_a_failed_stop_or_switch_off_frees_the_core(
    no_cyclic_gc: None,
    scenario: Scenario,
    kind: str,
    message: str,
) -> None:
    # The stop and safe-call paths held the error they raised (or a held log
    # line's argument) in a local of a frame on its own traceback (#148).
    core, microscope = _stand(
        declares=getattr(scenario, "declares", ()),
        engage_drives_z=getattr(scenario, "engage_drives_z", False),
        device_timeout_ms=getattr(scenario, "device_timeout_ms", None),
    )
    assert scenario(core, microscope) == (kind, message)
    core.failing.clear()
    watched = _watch(microscope, core)
    del microscope, core
    assert _alive(watched) == _FACADE_AND_CORE_FREED


# --- the Executor's own paths, and a second Ctrl-C --------------------------


class _SecondCtrlC(FakeCore):
    """A stand whose XY stop is interrupted, a fresh ``KeyboardInterrupt`` each time."""

    def stop(self, label: str) -> None:
        if label == "XY":
            raise KeyboardInterrupt("second Ctrl-C")
        super().stop(label)


def test_a_second_ctrl_c_in_the_emergency_stop_frees_the_core(
    no_cyclic_gc: None,
) -> None:
    # `_stop_stages` re-raises the interrupt once every stage has had its stop.
    core = _SecondCtrlC.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    assert _outcome(microscope.stop) == ("KeyboardInterrupt", "second Ctrl-C")
    assert "stop('Z')" in core.log  # the Z stop was sent after the interrupted XY one
    watched = _watch(microscope, core)
    del microscope, core
    assert _alive(watched) == _FACADE_AND_CORE_FREED


def _executor(logger: logging.Logger | None = None) -> Executor:
    return Executor(
        dry_run=False,
        lock=threading.RLock(),
        logger=logger or logging.getLogger("test.error-exits"),
    )


def _built(
    watched: list[weakref.ref[BaseException]], error: BaseException
) -> BaseException:
    """``error``, watched: for a raise that holds it in no local of the test."""
    watched.append(weakref.ref(error))
    return error


class _LinkedFailureError(RuntimeError):
    """A subclass: a built-in exception cannot be weakly referenced."""


class _LinkedInterrupt(KeyboardInterrupt):
    """A subclass: a built-in exception cannot be weakly referenced."""


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(_LinkedFailureError, id="failure"),
        pytest.param(_LinkedInterrupt, id="interrupt"),
    ],
)
def test_a_linked_stop_that_raises_still_stops_the_device_and_is_freed(
    no_cyclic_gc: None, raised: type[BaseException]
) -> None:
    # `_stop_linked` keeps the failure of a linked call, or the interrupt of
    # one, until the device's own stop was sent.
    executor, watched, stops = _executor(), [], []

    def linked() -> None:
        raise _built(watched, raised("linked stop"))

    z = Motion("Z", "z Z", lambda: False, lambda: stops.append("Z"))
    with executor.linked_stop("Z", linked):
        outcome = _outcome(lambda: executor.stop(z))
    assert outcome == (raised.__name__, "linked stop")
    assert stops == ["Z"]
    assert [ref() is not None for ref in watched] == [False]


class _StopRefusedError(RuntimeError):
    """A subclass: a built-in exception cannot be weakly referenced."""


class _InterruptsOnAFailedStop(logging.Handler):
    """A handler whose write of the line of a failed stop is cut by a second Ctrl-C."""

    def emit(self, record: logging.LogRecord) -> None:
        if "failed" in record.getMessage():
            raise KeyboardInterrupt("second Ctrl-C")


def test_a_second_ctrl_c_while_the_held_lines_are_written_frees_the_error(
    no_cyclic_gc: None,
) -> None:
    # A held line's arguments hold the error of the failed stop, and the list
    # is a local of `hold_log`'s frame: `_write_held` empties it in a `finally`.
    logging.disable(logging.NOTSET)  # the autouse fixture restores it
    logger = logging.getLogger("test.error-exits.interrupted-write")
    logger.propagate = False
    handler = _InterruptsOnAFailedStop()
    logger.addHandler(handler)
    try:
        executor, watched = _executor(logger), []

        def refuse() -> None:
            raise _built(watched, _StopRefusedError("stop refused"))

        xy = Motion("XY", "xy_stage XY", lambda: False, refuse)
        outcome = _outcome(lambda: executor.hold_log(lambda: executor.stop(xy)))
    finally:
        logger.removeHandler(handler)
    assert outcome == ("KeyboardInterrupt", "second Ctrl-C")
    assert [ref() is not None for ref in watched] == [False]


def test_a_refusal_of_a_compound_stopped_since_its_call_is_freed(
    no_cyclic_gc: None,
) -> None:
    # `refuse_if_stopped` takes the refusal as a parameter, and raises it.
    executor, watched = _executor(), []
    z = Motion("Z", "z Z", lambda: False, lambda: None)

    def run() -> None:
        executor.stop(z)  # after the compound was called
        executor.refuse_if_stopped("Z", _built(watched, HardwareError("stopped")))

    assert _outcome(lambda: executor.compound("z: move_to 3.0 µm", run)) == (
        "HardwareError",
        "stopped",
    )
    assert [ref() is not None for ref in watched] == [False]


# --- SampleCamera -----------------------------------------------------------


def _sample_camera(microscope: Microscope) -> SampleCamera:
    return SampleCamera(
        microscope, PlateSample(noise_std=0.0), pixel_size_um=1.0, shape=(32, 32)
    )


def test_a_sample_camera_override_frees_the_core(no_cyclic_gc: None) -> None:
    # The camera is a capability of the facade it holds: a strong reference
    # both ways was a cycle through the core (FM-92).
    core, microscope = _stand()
    microscope.override(Camera, _sample_camera(microscope))
    microscope.require(Camera).snap()
    watched = _watch(microscope, core)
    del microscope, core
    assert _alive(watched) == _FACADE_AND_CORE_FREED


def test_a_sample_camera_outliving_its_microscope_refuses_to_snap(
    no_cyclic_gc: None,
) -> None:
    core, microscope = _stand()
    camera = _sample_camera(microscope)
    core.log.clear()
    del microscope
    kind, message = _outcome(camera.snap)
    assert (kind, message) == (
        "HardwareError",
        "SampleCamera belongs to a Microscope that no longer exists; keep the "
        "Microscope alive for as long as the camera is in use",
    )
    assert core.log == []


# --- smc ui plate (m3 §15.3, #197) ---------------------------------------------


def _serve_returns(page: Any, *, port: int) -> None:
    """The operator's Ctrl-C, which ``serve`` takes: it returns."""


def _serve_ends_alone(page: Any, *, port: int) -> None:
    raise PageServerError("the plate page's server stopped on its own")


def _serve_has_a_bug(page: Any, *, port: int) -> None:
    raise RuntimeError("a bug in the page")


@pytest.mark.parametrize(
    ("serve", "code"),
    [
        pytest.param(_serve_returns, 0, id="ctrl-c"),
        pytest.param(_serve_ends_alone, 1, id="server-ended-alone"),
        pytest.param(_serve_has_a_bug, 1, id="bug"),
    ],
)
def test_smc_ui_plate_that_ends_frees_its_core(
    no_cyclic_gc: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    serve: Callable[..., None],
    code: int,
) -> None:
    # The page holds the facade and the stand-in `serve` is handed the page: neither
    # may outlive the command in a cycle (FM-92, FM-98). The page module is a stand-in,
    # so no Panel is needed.
    module = types.ModuleType("smc.ui.plate_page")
    module.__dict__["serve"] = serve
    monkeypatch.setitem(sys.modules, "smc.ui.plate_page", module)
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    args = ["ui", "plate", "--run-doc", str(tmp_path / "run.json"), "--port", str(port)]
    exit_code, watched = _run_cli(monkeypatch, FakeCore, args)
    assert exit_code == code
    assert _alive(watched) == _FACADE_AND_CORE_FREED
