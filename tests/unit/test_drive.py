"""The drive actions of ``smc.drive`` on ``FakeCore`` (m5 §6.1, #263).

Each action is one capability call, and its text is what the stand reads after
it, never what was asked (FM-155, FM-167): where a readback must disagree with
the request, a stub capability goes in through ``override``. Texts are asserted
whole (FM-45).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus

from smc import drive
from smc.drive import DriveResult
from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    FocusStep,
    LightSource,
    Limits,
    ObjectiveTurret,
    PropertyInfo,
    XYStage,
    ZStage,
)
from smc.hardware.errors import HardwareError, SafetyRefusedError
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.testing import FakeCore

TWENTY_X = "Nikon 20X Plan Fluor ELWD"
TEN_X = "Nikon 10X S Fluor"
SHUTTER_LIGHT = "White Light Shutter"


def _open(*, dry_run: bool, light: str | None = None) -> Microscope:
    profile = Profile.demo()
    if light is not None:
        profile = profile.model_copy(deep=True)
        profile.roles.assign[Role.light_source] = light
    return Microscope.from_core(
        cast("CMMCorePlus", FakeCore.demo_like()), profile, dry_run=dry_run
    )


@pytest.fixture
def dry() -> Iterator[Microscope]:
    microscope = _open(dry_run=True)
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def shutter_light_dry() -> Iterator[Microscope]:
    microscope = _open(dry_run=True, light=SHUTTER_LIGHT)
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def shutter_light() -> Iterator[Microscope]:
    microscope = _open(dry_run=False, light=SHUTTER_LIGHT)
    try:
        yield microscope
    finally:
        microscope.close()


def _fake(microscope: Microscope) -> FakeCore:
    return cast("FakeCore", microscope.core)


# --- stubs whose readback differs from the request --------------------------------


class LandsElsewhere:
    """An XY stage whose every move lands at ``(-0.001, 99.996)``, whatever was asked."""

    landed = XY(-0.001, 99.996)

    def position_um(self) -> XY:
        return self.landed

    def move_to_um(self, x_um: float, y_um: float) -> XY:
        return self.landed

    def move_by_um(self, dx_um: float, dy_um: float, *, force: bool = False) -> XY:
        return self.landed

    def wait(self, timeout_s: float | None = None) -> None:
        return None

    def is_busy(self) -> bool:
        return False

    def stop(self) -> None:
        return None

    def limits_um(self) -> tuple[Limits, Limits] | None:
        return None


class StubTurret:
    """A turret with the labels it is given, whose ``current`` is ``stays_at`` after any turn."""

    def __init__(self, labels: list[str], stays_at: str) -> None:
        self._labels = labels
        self._current = stays_at
        self.selected: list[tuple[str, bool]] = []

    def labels(self) -> list[str]:
        return list(self._labels)

    def current(self) -> str:
        return self._current

    def select(self, label: str, *, confirm: bool = False) -> str:
        self.selected.append((label, confirm))
        return label

    def pixel_size_um(self, label: str | None = None) -> float:
        return 0.0


class NeverLocks:
    """An autofocus whose ``engage`` returns ``False``, as one that found no lock does."""

    def is_engaged(self) -> bool:
        return False

    def is_locked(self) -> bool:
        return False

    def is_in_range(self) -> bool | None:
        return None

    def engage(self, timeout_s: float | None = None) -> bool:
        return False

    def disengage(self) -> None:
        return None

    def offset_native(self) -> float:
        return 0.0

    def set_offset_native(self, value: float) -> float:
        return value

    def offset_by_native(self, delta: float, *, force: bool = False) -> float:
        return delta

    def offset_unit(self) -> str:
        return ""

    def offset_limits_native(self) -> tuple[float, float] | None:
        return None

    def focus_by_um(self, dz_um: float, *, force: bool = False) -> FocusStep:
        raise NotImplementedError


class StuckLight:
    """A light whose ``set_on`` returns what it was asked and whose ``is_on`` reads ``stuck``."""

    def __init__(self, stuck: bool) -> None:
        self._stuck = stuck

    def device(self) -> str:
        return "Stuck"

    def intensity_knob(self) -> PropertyInfo | None:
        return None

    def intensity_native(self) -> float | None:
        return None

    def set_intensity_native(self, value: float) -> float:
        return value

    def is_on(self) -> bool | None:
        return self._stuck

    def set_on(self, on: bool) -> bool:
        return on


# --- texts ----------------------------------------------------------------------------


def test_jog_xy_text_is_the_readback_with_minus_zero_as_zero(
    fake_microscope: Microscope,
) -> None:
    fake_microscope.override(XYStage, LandsElsewhere())

    result = drive.jog_xy(fake_microscope, 100.0, 100.0)

    # Where the stage says it landed, not (100, 100); -0.001 shows as 0.00.
    assert result == DriveResult("XY (0.00, 100.00) µm")


def test_jog_z_text_is_the_position_the_drive_returns(
    fake_microscope: Microscope,
) -> None:
    fake_microscope.require(ZStage).move_to_um(-7.5)

    assert drive.jog_z(fake_microscope, 2.5) == DriveResult("Z -5.00 µm")


def test_the_formatters_show_minus_zero_as_zero() -> None:
    assert drive.um_text(-0.004) == "0.00"
    assert drive.um_text(-0.005) == "-0.01"
    assert drive.xy_text(XY(-0.0, 1.0)) == "XY (0.00, 1.00) µm"
    assert drive.z_text(-0.0) == "Z 0.00 µm"


_DRY_RUN_CASES: list[tuple[str, Callable[[Microscope], DriveResult], str]] = [
    (
        "jog_xy",
        lambda m: drive.jog_xy(m, 10.0, -20.0),
        "[dry-run] would move to XY (10.00, -20.00) µm; the stage is at XY (0.00, 0.00) µm",
    ),
    (
        "jog_z",
        lambda m: drive.jog_z(m, 5.0),
        "[dry-run] would move to Z 5.00 µm; the drive is at Z 0.00 µm",
    ),
    (
        "set_objective",
        lambda m: drive.set_objective(m, TWENTY_X, confirm=True),
        f"[dry-run] would turn to '{TWENTY_X}'; the turret is at '{TEN_X}'",
    ),
    (
        "engage_autofocus",
        drive.engage_autofocus,
        "[dry-run] would engage the autofocus; autofocus off",
    ),
]


@pytest.mark.parametrize(
    ("action", "expected"),
    [(case[1], case[2]) for case in _DRY_RUN_CASES],
    ids=[case[0] for case in _DRY_RUN_CASES],
)
def test_each_action_in_dry_run_says_what_it_would_do_and_the_stand_stays(
    dry: Microscope, action: Callable[[Microscope], DriveResult], expected: str
) -> None:
    _fake(dry).log.clear()

    result = action(dry)

    assert result == DriveResult(expected, reached=True)
    state = dry.state()
    assert state.xy == XY(0.0, 0.0)
    assert state.z_um == 0.0
    assert state.objective == TEN_X
    assert state.autofocus_engaged is False
    sent = [
        line
        for line in _fake(dry).log
        if line.startswith(
            (
                "setXYPosition",
                "setPosition",
                "setRelative",
                "setStateLabel",
                "enableContinuousFocus",
            )
        )
    ]
    assert sent == []


def test_switching_off_is_sent_in_dry_run_and_says_so(
    dry: Microscope, shutter_light_dry: Microscope
) -> None:
    # Dry run sends no "on", so the stand is put on through its core, as an operator
    # at the stand's own controls would have left it.
    _fake(dry).enableContinuousFocus(True)
    assert dry.require(Autofocus).is_engaged()
    _fake(shutter_light_dry).setShutterOpen(SHUTTER_LIGHT, True)
    assert shutter_light_dry.require(LightSource).is_on() is True

    autofocus = drive.disengage_autofocus(dry)
    light = drive.switch_light(shutter_light_dry, False)

    assert autofocus == DriveResult(
        "autofocus off (dry run: switching off is sent all the same)"
    )
    assert not dry.require(Autofocus).is_engaged()
    assert light == DriveResult(
        "light off (dry run: switching off is sent all the same)"
    )
    assert shutter_light_dry.require(LightSource).is_on() is False


def test_switching_the_light_on_in_dry_run_sends_nothing_and_says_what_it_reads(
    shutter_light_dry: Microscope,
) -> None:
    # FM-47: the demo's shutter may read open on an open; the state is set first.
    _fake(shutter_light_dry).setShutterOpen(SHUTTER_LIGHT, False)

    result = drive.switch_light(shutter_light_dry, True)

    assert result == DriveResult("[dry-run] would switch the light on; it is off")
    assert shutter_light_dry.require(LightSource).is_on() is False


def test_set_objective_without_confirm_is_refused_and_the_turret_stays(
    fake_microscope: Microscope,
) -> None:
    _fake(fake_microscope).log.clear()

    with pytest.raises(SafetyRefusedError) as refused:
        drive.set_objective(fake_microscope, TWENTY_X, confirm=False)

    assert refused.value.how_to_force == "pass confirm=True once the stage is clear"
    assert fake_microscope.require(ObjectiveTurret).current() == TEN_X
    assert not any(
        line.startswith("setStateLabel") for line in _fake(fake_microscope).log
    )


def test_set_objective_with_confirm_reads_back_the_current_label(
    fake_microscope: Microscope,
) -> None:
    result = drive.set_objective(fake_microscope, TWENTY_X, confirm=True)

    assert result == DriveResult(f"objective '{TWENTY_X}'")
    assert fake_microscope.require(ObjectiveTurret).current() == TWENTY_X


def test_set_objective_with_an_unknown_label_raises_the_facades_value_error(
    fake_microscope: Microscope,
) -> None:
    with pytest.raises(ValueError, match="has no position 'Nikon 100X'"):
        drive.set_objective(fake_microscope, "Nikon 100X", confirm=True)
    assert fake_microscope.require(ObjectiveTurret).current() == TEN_X


def test_a_turret_that_stops_short_of_the_label_is_not_reached(
    fake_microscope: Microscope,
) -> None:
    turret = StubTurret([TEN_X, TWENTY_X], stays_at=TEN_X)
    fake_microscope.override(ObjectiveTurret, turret)

    result = drive.set_objective(fake_microscope, TWENTY_X, confirm=True)

    assert turret.selected == [(TWENTY_X, True)]
    assert result == DriveResult(
        f"objective '{TEN_X}', although '{TWENTY_X}' was asked", reached=False
    )


def test_an_engage_that_does_not_lock_is_not_reached(
    fake_microscope: Microscope,
) -> None:
    fake_microscope.override(Autofocus, NeverLocks())

    result = drive.engage_autofocus(fake_microscope)

    assert result == DriveResult("autofocus did not lock; it is off", reached=False)


def test_an_engage_that_locks_says_so(fake_microscope: Microscope) -> None:
    result = drive.engage_autofocus(fake_microscope)

    assert result == DriveResult("autofocus locked")
    assert fake_microscope.require(Autofocus).is_locked()


def test_a_disengage_reads_the_autofocus_after_it(fake_microscope: Microscope) -> None:
    fake_microscope.require(Autofocus).engage()

    assert drive.disengage_autofocus(fake_microscope) == DriveResult("autofocus off")
    assert not fake_microscope.require(Autofocus).is_engaged()


@pytest.mark.parametrize(
    ("on", "stuck", "expected"),
    [
        (True, False, "light off, although on was sent"),
        (False, True, "light on, although off was sent"),
    ],
)
def test_a_light_whose_readback_disagrees_is_not_reached(
    fake_microscope: Microscope, on: bool, stuck: bool, expected: str
) -> None:
    fake_microscope.override(LightSource, StuckLight(stuck))

    result = drive.switch_light(fake_microscope, on)

    assert result == DriveResult(expected, reached=False)


def test_a_shutter_typed_light_switches_and_reads_back(
    shutter_light: Microscope,
) -> None:
    _fake(shutter_light).setShutterOpen(SHUTTER_LIGHT, False)

    assert drive.switch_light(shutter_light, True) == DriveResult("light on")
    assert shutter_light.require(LightSource).is_on() is True
    assert drive.switch_light(shutter_light, False) == DriveResult("light off")
    assert shutter_light.require(LightSource).is_on() is False


def test_an_empty_turret_label_reads_as_two_quotes(
    fake_microscope: Microscope,
) -> None:
    # FM-125: an empty position is a label too, and must be visible as one.
    fake_microscope.override(ObjectiveTurret, StubTurret(["", TEN_X], stays_at=""))

    result = drive.set_objective(fake_microscope, "", confirm=True)

    assert result == DriveResult("objective ''")


def test_drive_reaches_the_stand_only_through_the_facade() -> None:
    # The module's imports: smc.hardware and the standard library, nothing of the UI or
    # the CLI, and no Micro-Manager.
    import ast
    from pathlib import Path

    source = Path(drive.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported == {
        "__future__",
        "dataclasses",
        "typing",
        "smc.hardware.capabilities",
        "smc.hardware.microscope",
    }


class StaysEngaged(NeverLocks):
    """An autofocus that reads engaged and locked whatever ``disengage`` sent."""

    def is_engaged(self) -> bool:
        return True

    def is_locked(self) -> bool:
        return True


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_a_disengage_that_leaves_the_autofocus_on_is_not_reached(
    dry_run: bool,
) -> None:
    # Switching off is sent in dry run too, so what the stand reads after it is a
    # readback in both modes.
    microscope = _open(dry_run=dry_run)
    try:
        microscope.override(Autofocus, StaysEngaged())
        result = drive.disengage_autofocus(microscope)
    finally:
        microscope.close()
    suffix = " (dry run: switching off is sent all the same)" if dry_run else ""
    assert result == DriveResult(f"autofocus locked{suffix}", reached=False)


class EngagesButLeavesItOn(StaysEngaged):
    """``engage`` returns ``False``, as with no lock, yet the autofocus still reads locked."""


def test_an_engage_that_returns_false_says_what_the_stand_reads(
    fake_microscope: Microscope,
) -> None:
    # FM-155: "it is off" is what the contract promises for False, not a readback.
    fake_microscope.override(Autofocus, EngagesButLeavesItOn())

    result = drive.engage_autofocus(fake_microscope)

    assert result == DriveResult(
        "autofocus did not lock; autofocus locked", reached=False
    )


class ReadsFail(NeverLocks):
    """An autofocus whose state cannot be read."""

    def is_engaged(self) -> bool:
        raise HardwareError("autofocus 'PFS': cannot read")

    def is_locked(self) -> bool:
        raise HardwareError("autofocus 'PFS': cannot read")


def test_an_autofocus_that_cannot_be_read_says_why(
    fake_microscope: Microscope,
) -> None:
    fake_microscope.override(Autofocus, ReadsFail())

    result = drive.disengage_autofocus(fake_microscope)

    # The cause once, although both reads failed with it.
    assert result == DriveResult(
        "autofocus ? (autofocus: HardwareError: autofocus 'PFS': cannot read)",
        reached=False,
    )
    assert drive.autofocus_text(fake_microscope) == result.text
