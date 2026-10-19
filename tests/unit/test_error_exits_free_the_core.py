"""An error exit must free the core by reference counting alone (#148, FM-92).

The cyclic collector runs at an arbitrary moment, and at the exit of a command
during interpreter finalization, where a thread still inside MMCore aborts the
process (#142). #142 made the success paths free the stand; the error paths
kept it alive, because an exception that is kept where its own traceback
reaches it is a cycle: exception, traceback, frame, local, exception.

Every test here runs under ``no_cyclic_gc`` and watches the facade and the core
with weak references. A new error path of a stop, a safe call or a CLI exit
goes in here. Hazards, measured on #148:

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

import weakref
from collections.abc import Callable
from typing import Any, cast

import pytest
from pymmcore_plus import CMMCorePlus

import smc.cli as cli
from smc.hardware import Microscope
from smc.hardware.capabilities import Autofocus, Camera, XYStage, ZStage
from smc.hardware.profile import Profile
from smc.testing import FakeCore, PlateSample, SampleCamera

_FACADE_AND_CORE_FREED = {"facade": False, "core": False}


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
    except Exception as exc:
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
    # `_reported` was a generator: one that converts an exception keeps the
    # command's frames, and the stand, in a cycle on Python 3.11+ (#148).
    exit_code, watched = _run_cli(monkeypatch, core_type, args)
    assert exit_code == code
    assert _alive(watched) == _FACADE_AND_CORE_FREED


def test_a_cli_command_that_succeeds_frees_its_core(
    no_cyclic_gc: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pins that the mechanism of the tests above frees a core at all.
    exit_code, watched = _run_cli(monkeypatch, FakeCore, ["stage", "get"])
    assert exit_code == 0
    assert _alive(watched) == _FACADE_AND_CORE_FREED


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
