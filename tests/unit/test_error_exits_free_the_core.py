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
import threading
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import tifffile
import typer
from pymmcore_plus import CMMCorePlus

import smc.cli as cli
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, XYStage, ZStage
from smc.hardware.errors import HardwareError
from smc.hardware.profile import Profile
from smc.hardware.safety import Executor, Motion
from smc.testing import FakeCore, PlateSample, SampleCamera

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


def _stand(*, dry_run: bool = False) -> tuple[FakeCore, Microscope]:
    core = FakeCore.demo_like()
    return core, Microscope.from_core(
        cast("CMMCorePlus", core), Profile.demo(), dry_run=dry_run
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
    monkeypatch: pytest.MonkeyPatch, core_type: type[FakeCore], args: list[str]
) -> tuple[int | str | None, dict[str, weakref.ref[Any]]]:
    """Run ``smc <args>`` on a fake demo stand; the exit code and what to watch.

    Through ``cli.app``, not ``main()`` (its collection would hide the cycle)
    and not ``CliRunner`` (it adds a cycle of its own).
    """
    watched: dict[str, weakref.ref[Any]] = {}

    def open_stand(
        cls: type[Microscope], profile: str = "demo", *, dry_run: bool = False
    ) -> Microscope:
        core = core_type.demo_like()
        microscope = Microscope.from_core(
            cast("CMMCorePlus", core), Profile.demo(), dry_run=dry_run
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
    ],
)
def test_a_failed_stop_or_switch_off_frees_the_core(
    no_cyclic_gc: None,
    scenario: Callable[[FakeCore, Microscope], tuple[str, str]],
    kind: str,
    message: str,
) -> None:
    # The stop and safe-call paths held the error they raised (or a held log
    # line's argument) in a local of a frame on its own traceback (#148).
    core, microscope = _stand()
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
