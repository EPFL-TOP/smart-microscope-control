"""Contract fixtures: one stand per backend, its envelope, and the put-back (design §8).

Every contract runs on three backends: ``fake`` (``FakeCore``), ``demo``
(Micro-Manager's demo devices) and ``hardware`` (the stand ``--profile``
names; deselected unless ``-m hardware``). The tests never import a
backend class: they only go through ``Microscope``.

**What a contract may move.** ``envelope`` records where the stand started
and replaces the profile's ``[safety]`` with limits around that start: XY
± 100 µm, Z ± 3 µm, a 50 µm XY jog limit and a 1 µm Z jog limit. Every
target a contract *sends* is within 60 µm in XY, and in Z at most 2 µm and
only below the start, which is away from the sample on an inverted stand.
The autofocus offset gets soft limits at its start ± 2.0 and a jog limit of
1.0, in its own units (m2 §3.13); a contract moves it at most 1.0 from the
start, with the autofocus off, except in the locked ``focus_by_um`` case.
The offset is also sent in whole units only: the ``fake`` stand's autofocus
holds whole units from 7927, like the Ti2-E's reads, so every offset a
contract sends is the start plus a whole number (FM-119, #207). Every target
outside the envelope is one the layer must refuse, so none of them is ever
sent. The ``microscope`` fixture then puts the stand back where it started.

**On ``hardware``, the turret never turns, and no capture is ever run**
(m2 §8.8): a turn can drive an objective into the plate, and a capture
writes the stand's ``.cfg``. A select of the current objective (which sends
nothing) and of the current channel are the only selections made there.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Autofocus,
    Camera,
    Channels,
    LightSource,
    ObjectiveTurret,
    Properties,
    Shutter,
    XYStage,
    ZStage,
)
from smc.hardware.profile import Profile, SafetySection

if TYPE_CHECKING:
    from smc.testing import FakeCore

T = TypeVar("T")

#: The fixture that opens each backend's stand; it owns the core and closes it.
_STANDS = {
    "fake": "whole_unit_fake_microscope",
    "demo": "demo_microscope",
    "hardware": "hardware_microscope",
}


@dataclass(frozen=True)
class Envelope:
    """Where the stand started, and the ``[safety]`` the contracts run under.

    A start value is ``None`` when the stand lacks that capability. Tests
    read the numbers from here, never from constants of their own.
    """

    xy: XY | None
    z_um: float | None
    exposure_ms: float | None
    shutter_open: bool | None
    auto_shutter: bool | None
    autofocus_engaged: bool | None = None
    #: In the offset device's own units.
    offset_native: float | None = None
    objective: str | None = None
    #: ``None`` also when the stand matched no preset: then it is not put back.
    channel: str | None = None
    max_jog_um: float = 50.0
    #: Below the 2 µm a contract may move Z, so a forced jog stays inside the envelope.
    max_z_jog_um: float = 1.0
    xy_span_um: float = 100.0
    z_span_um: float = 3.0
    tolerance_um: float = 0.5
    #: The autofocus offset's, in its own units (m2 §3.13). The tolerance stays
    #: 0.5 although the stand holds whole units: every delta a contract sends
    #: (1.0, 2.0, 3.0) differs from it and from the limit it is compared with,
    #: and a wider one would let a jog clamped to the limit pass (#207).
    max_offset_jog: float = 1.0
    offset_span: float = 2.0
    offset_tolerance: float = 0.5

    def profile(self, base: Profile) -> Profile:
        """``base`` with soft limits around the start and the contract's jog limits.

        Every jog limit is set here: a stand profile's own ``max_z_jog_um``
        (50 µm, say) would otherwise survive the merge and the contract's
        1.5 µm refusal case would pass the guard on hardware; the same holds
        for ``max_autofocus_offset_jog``. Every other key of ``base`` is kept,
        and so is ``source``.

        The window must lie inside the profile's own soft limits, which exist
        for a reason (the stage's travel, an obstacle): a stand that starts
        too close to one fails the test before anything moves, rather than
        have the contracts widen the limit.
        """
        own = base.safety
        xy_limits = None
        if self.xy is not None:
            x0, y0, span = self.xy.x_um, self.xy.y_um, self.xy_span_um
            xy_limits = ((x0 - span, x0 + span), (y0 - span, y0 + span))
            if own.xy_soft_limits_um is not None:
                for axis, window, limit in zip(
                    "XY", xy_limits, own.xy_soft_limits_um, strict=True
                ):
                    _require_inside(axis, window, limit)
        z_limits = None
        if self.z_um is not None:
            z_limits = (self.z_um - self.z_span_um, self.z_um + self.z_span_um)
            if own.z_soft_limits_um is not None:
                _require_inside("Z", z_limits, own.z_soft_limits_um)
        offset_limits = None
        if self.offset_native is not None:
            o0, span = self.offset_native, self.offset_span
            offset_limits = (o0 - span, o0 + span)
            if own.autofocus_offset_soft_limits is not None:
                # FM-48: inside the profile's own limits, in its own units.
                _require_inside(
                    "autofocus offset",
                    offset_limits,
                    own.autofocus_offset_soft_limits,
                    unit=base.autofocus.offset_unit or "device units",
                )
        safety = SafetySection.model_validate(
            {
                **base.safety.model_dump(),
                "max_jog_um": self.max_jog_um,
                "max_z_jog_um": self.max_z_jog_um,
                "xy_soft_limits_um": xy_limits,
                "z_soft_limits_um": z_limits,
                "max_autofocus_offset_jog": self.max_offset_jog,
                "autofocus_offset_soft_limits": offset_limits,
            }
        )
        return base.model_copy(update={"safety": safety})


def _require_inside(
    axis: str,
    window: tuple[float, float],
    limit: tuple[float, float],
    *,
    unit: str = "µm",
) -> None:
    """Fail the test unless the contract's ``window`` lies within the profile's ``limit``."""
    if window[0] < limit[0] or window[1] > limit[1]:
        pytest.fail(
            f"the stand starts too close to its {axis} soft limits "
            f"[{limit[0]:g}, {limit[1]:g}] {unit}: the contracts need "
            f"[{window[0]:g}, {window[1]:g}] {unit}. Move it towards the middle "
            f"of its travel and run them again"
        )


@pytest.fixture(
    params=[
        pytest.param("fake", id="fake"),
        pytest.param("demo", id="demo", marks=pytest.mark.demo),
        pytest.param("hardware", id="hardware", marks=pytest.mark.hardware),
    ]
)
def backend(request: pytest.FixtureRequest) -> str:
    """Which implementation the contract runs on."""
    return str(request.param)


@pytest.fixture
def whole_unit_fake_microscope(
    fake_core: FakeCore, request: pytest.FixtureRequest
) -> Microscope:
    """``fake_microscope`` over an autofocus that holds whole units, from 7927.

    The Ti2-E's ``PFSOffset`` reads whole numbers (7927, 7937, 9370, ...), and
    a start of ``0.0`` would not show a contract that sends fractions: from
    there, truncation towards zero errs the other way (#207). The device is
    configured before the facade is built, which reads the offset.
    """
    fake_core.hold_whole_units(fake_core.getAutoFocusDevice())
    fake_core.setAutoFocusOffset(7927.0)
    stand: Microscope = request.getfixturevalue("fake_microscope")
    return stand


@pytest.fixture
def stand(backend: str, request: pytest.FixtureRequest) -> Microscope:
    """The backend's facade on its own profile; its fixture skips, fails and closes."""
    stand: Microscope = request.getfixturevalue(_STANDS[backend])
    return stand


@pytest.fixture
def envelope(stand: Microscope) -> Envelope:
    """Where ``stand`` is now, read through it before any contract moves it."""
    xy = stand.get(XYStage)
    z = stand.get(ZStage)
    camera = stand.get(Camera)
    shutter = stand.get(Shutter)
    autofocus = stand.get(Autofocus)
    turret = stand.get(ObjectiveTurret)
    channels = stand.get(Channels)
    return Envelope(
        xy=None if xy is None else xy.position_um(),
        z_um=None if z is None else z.position_um(),
        exposure_ms=None if camera is None else camera.exposure_ms(),
        shutter_open=None if shutter is None else shutter.is_open(),
        auto_shutter=None if shutter is None else shutter.auto_shutter(),
        autofocus_engaged=None if autofocus is None else autofocus.is_engaged(),
        offset_native=None if autofocus is None else autofocus.offset_native(),
        objective=None if turret is None else turret.current(),
        channel=None if channels is None else channels.current(),
    )


@pytest.fixture
def microscope(
    stand: Microscope, envelope: Envelope, request: pytest.FixtureRequest
) -> Iterator[Microscope]:
    """A second facade over ``stand``'s core, under the envelope's ``[safety]``.

    Never closed: ``stand`` owns the core. Its teardown puts the stand back,
    unless the run was interrupted: see ``_interrupted``.
    """
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    yield facade
    try:
        _put_back(facade, envelope, interrupted=_interrupted(request.session))
    except PutBackInterruptedError as exc:
        # The operator's stop must end the run, or the next test reopens the
        # stand and moves it. Re-raising the KeyboardInterrupt would end it
        # too, but pytest then drops this test's remaining finalizers, the
        # stand's close() among them; shouldstop ends the run after them.
        request.session.shouldstop = f"Ctrl-C during the put-back: {exc}"
        raise


def _interrupted(session: pytest.Session) -> bool:
    """Whether the run is being torn down after a Ctrl-C (or ``pytest.exit``).

    An operator who presses Ctrl-C during a move has stopped the stand on
    purpose. The layer sends the stop and drops the motion once the stage
    reads idle, without halting (``safety._give_up``), so the facade looks
    like any finished test's. pytest still runs fixture teardowns after a
    ``KeyboardInterrupt``, from ``pytest_sessionfinish``, by which time it has
    set ``exitstatus`` to ``INTERRUPTED``: that, not the facade, is what tells
    the put-back not to move the stand again.

    ``session.shouldstop`` is deliberately not read: ``--stepwise`` sets it
    after an ordinary failure, and that stand should still be put back.
    """
    return bool(session.exitstatus == pytest.ExitCode.INTERRUPTED)


@pytest.fixture
def dry_microscope(stand: Microscope, envelope: Envelope) -> Microscope:
    """The same facade in dry-run; nothing to put back, since it sends no mutation."""
    return Microscope.from_core(
        stand.core, envelope.profile(stand.profile), dry_run=True
    )


def _capability(facade: Microscope, backend: str, capability: type[T]) -> T:
    """``require(capability)``; on ``hardware`` only, a stand without it skips.

    On ``fake`` and ``demo`` a missing capability is a broken role
    resolution, which must fail the test rather than skip it.
    """
    if backend == "hardware" and not facade.has(capability):
        pytest.skip(f"this stand has no {capability.__name__}")
    return facade.require(capability)


@pytest.fixture
def xy(microscope: Microscope, backend: str) -> XYStage:
    """The live XY stage."""
    return _capability(microscope, backend, XYStage)


@pytest.fixture
def z(microscope: Microscope, backend: str) -> ZStage:
    """The live Z drive."""
    return _capability(microscope, backend, ZStage)


@pytest.fixture
def camera(microscope: Microscope, backend: str) -> Camera:
    """The live camera."""
    return _capability(microscope, backend, Camera)


@pytest.fixture
def shutter(microscope: Microscope, backend: str) -> Shutter:
    """The live shutter."""
    return _capability(microscope, backend, Shutter)


@pytest.fixture
def properties(microscope: Microscope, backend: str) -> Properties:
    """Every device's properties."""
    return _capability(microscope, backend, Properties)


@pytest.fixture
def autofocus(microscope: Microscope, backend: str) -> Autofocus:
    """The live autofocus, its lock read once.

    On a stand that declares ``engage_drives_z_to_offset``, the layer engages
    only at an offset it has seen locked (m2 §12.2 D1): the read makes the
    operator's lock, taken at the stand before the run, a seen one.
    """
    autofocus = _capability(microscope, backend, Autofocus)
    autofocus.is_locked()
    return autofocus


@pytest.fixture
def objective_turret(microscope: Microscope, backend: str) -> ObjectiveTurret:
    """The live objective turret; on ``hardware``, it is never turned."""
    return _capability(microscope, backend, ObjectiveTurret)


@pytest.fixture
def light_source(microscope: Microscope, backend: str) -> LightSource:
    """The live light source."""
    return _capability(microscope, backend, LightSource)


@pytest.fixture
def channels(microscope: Microscope, backend: str) -> Channels:
    """The live channels; on ``hardware``, only the current one is selected."""
    return _capability(microscope, backend, Channels)


@pytest.fixture
def dry_objective_turret(dry_microscope: Microscope, backend: str) -> ObjectiveTurret:
    """The objective turret in dry-run."""
    return _capability(dry_microscope, backend, ObjectiveTurret)


@pytest.fixture
def dry_channels(dry_microscope: Microscope, backend: str) -> Channels:
    """The channels in dry-run."""
    return _capability(dry_microscope, backend, Channels)


@pytest.fixture
def dry_autofocus(dry_microscope: Microscope, backend: str) -> Autofocus:
    """The autofocus in dry-run."""
    return _capability(dry_microscope, backend, Autofocus)


@pytest.fixture
def dry_xy(dry_microscope: Microscope, backend: str) -> XYStage:
    """The XY stage in dry-run."""
    return _capability(dry_microscope, backend, XYStage)


@pytest.fixture
def dry_z(dry_microscope: Microscope, backend: str) -> ZStage:
    """The Z drive in dry-run."""
    return _capability(dry_microscope, backend, ZStage)


class PutBack(Protocol):
    """The signature of ``_put_back``, for the tests that call it directly."""

    def __call__(
        self, facade: Microscope, envelope: Envelope, *, interrupted: bool = False
    ) -> None: ...


@pytest.fixture
def put_back() -> PutBack:
    """The put-back ``microscope`` runs at teardown, for the tests of it."""
    return _put_back


def _put_back(
    facade: Microscope, envelope: Envelope, *, interrupted: bool = False
) -> None:
    """Return the stand to ``envelope``, or stop it if it must not move again.

    A halted facade, one with a motion still registered, or an interrupted
    run is stopped instead: the stand's own ``close()`` cannot see this
    facade's motions, and an operator's Ctrl-C means "stop", not "go back".
    The light is still cut if it started off. Otherwise every item that
    differs is put back, light off first and on last, and the autofocus off
    while anything moves back (m2 §3.13): XY, Z, the objective (m2 §8.8), then
    the offset, which with the autofocus off moves only the offset lens. The
    channel comes back after the exposure. It is engaged again at the
    end if it started engaged, and a lock that does not come is a failure
    named like any other.

    Every step is attempted even if an earlier one failed, and one error
    then names every step that failed: at a stand, the operator must learn
    each item that was not put back, not only the first. A Ctrl-C during the
    put-back leaves only the stop and the shutter close to run (``_run_all``).
    """
    shutter = facade.get(Shutter)
    steps: list[tuple[str, Callable[[], None]]] = []
    if interrupted or _has_motion(facade):
        steps.append(("stop", facade.stop))
        if shutter is not None and envelope.shutter_open is False:
            steps.append(("shutter close", lambda: _close_shutter(shutter)))
        _run_all(steps)
        return

    tolerance_um = envelope.tolerance_um
    if shutter is not None and envelope.shutter_open is False:
        steps.append(("shutter close", lambda: _close_shutter(shutter)))
    autofocus = facade.get(Autofocus)
    if autofocus is not None:
        steps.append(("autofocus off", autofocus.disengage))
    xy = facade.get(XYStage)
    if xy is not None and (start := envelope.xy) is not None:
        steps.append(("xy", lambda: _put_back_xy(xy, start, tolerance_um)))
    z = facade.get(ZStage)
    if z is not None and (z0 := envelope.z_um) is not None:
        steps.append(("z", lambda: _put_back_z(z, z0, tolerance_um)))
    turret = facade.get(ObjectiveTurret)
    if turret is not None and (objective := envelope.objective) is not None:
        # After Z, which is back below its start; the autofocus is off already.
        steps.append(("objective", lambda: _put_back_objective(turret, objective)))
    if autofocus is not None and (o0 := envelope.offset_native) is not None:
        tolerance = envelope.offset_tolerance
        steps.append(("offset", lambda: _put_back_offset(autofocus, o0, tolerance)))
    camera = facade.get(Camera)
    if camera is not None and (ms := envelope.exposure_ms) is not None:
        steps.append(("exposure", lambda: _put_back_exposure(camera, ms)))
    channels = facade.get(Channels)
    if channels is not None and (channel := envelope.channel) is not None:
        steps.append(("channel", lambda: _put_back_channel(channels, channel)))
    if shutter is not None and (auto := envelope.auto_shutter) is not None:
        steps.append(("auto-shutter", lambda: _put_back_auto_shutter(shutter, auto)))
    if autofocus is not None and envelope.autofocus_engaged is True:
        steps.append(("autofocus on", lambda: _engage_again(autofocus)))
    if shutter is not None and envelope.shutter_open is True:
        steps.append(("shutter open", lambda: _open_shutter(shutter)))
    _run_all(steps)


def _has_motion(facade: Microscope) -> bool:
    """Whether the facade is halted or still has a motion registered."""
    state = facade.state()
    return state.halted or bool(state.moving)


#: The steps still run after a Ctrl-C during the put-back: they stop the
#: stand, cut the light and switch the autofocus off, which is what the
#: operator asked for; none of them can create motion.
_SAFE_STEPS = frozenset({"stop", "shutter close", "autofocus off"})


class PutBackInterruptedError(RuntimeError):
    """A Ctrl-C landed in a put-back step: the operator asked to stop.

    A ``RuntimeError``, so that pytest reports it as a teardown error and
    still runs the remaining finalizers; the ``microscope`` fixture then
    ends the session (``shouldstop``), so that no later test moves the stand.
    """


def _run_all(steps: list[tuple[str, Callable[[], None]]]) -> None:
    """Run every step; then raise one error naming each that failed, from the first.

    A Ctrl-C inside a step (a second one, when the run was already
    interrupted) is a failure of that step, not the end of the put-back:
    left to propagate, it would skip the shutter close queued after a stop
    and leave the light on. From then on only the safe steps run; the
    others are named as not attempted, so the operator knows what to check.
    The error is then a ``PutBackInterruptedError``, so that the run is not
    carried on as after an ordinary failure.
    """
    failed: list[tuple[str, BaseException]] = []
    skipped: list[str] = []
    ctrl_c = False
    for name, step in steps:
        if ctrl_c and name not in _SAFE_STEPS:
            skipped.append(name)
            continue
        try:
            step()
        except KeyboardInterrupt as exc:
            ctrl_c = True
            failed.append((name, exc))
        except Exception as exc:
            failed.append((name, exc))
    if failed:
        listed = "; ".join(f"{name}: {exc!r}" for name, exc in failed)
        if skipped:
            listed += f"; not attempted after Ctrl-C: {', '.join(skipped)}"
        error = PutBackInterruptedError if ctrl_c else RuntimeError
        raise error(
            f"the stand was not put back where it started ({listed}); check it "
            f"before the next run"
        ) from failed[0][1]


def _close_shutter(shutter: Shutter) -> None:
    """Cut the light without reading first: a failed read must not leave it on.

    Closing is never refused (§13), and closing a closed shutter is harmless.
    """
    shutter.set_open(False)


def _open_shutter(shutter: Shutter) -> None:
    """Open only if closed: a failed read leaves the light off, the safe side."""
    if not shutter.is_open():
        shutter.set_open(True)


def _put_back_xy(xy: XYStage, start: XY, tolerance_um: float) -> None:
    here = xy.position_um()
    if max(abs(here.x_um - start.x_um), abs(here.y_um - start.y_um)) > tolerance_um:
        xy.move_to_um(start.x_um, start.y_um)


def _put_back_z(z: ZStage, z0_um: float, tolerance_um: float) -> None:
    # The autofocus was switched off before; engaging it is the last step's.
    if abs(z.position_um() - z0_um) > tolerance_um:
        z.move_to_um(z0_um, keep_autofocus=False)


def _put_back_objective(turret: ObjectiveTurret, start: str) -> None:
    """Turn back only if a contract turned it: on ``hardware`` none does."""
    if turret.current() != start:
        turret.select(start, confirm=True)


def _put_back_channel(channels: Channels, start: str) -> None:
    if channels.current() != start:
        channels.select(start)


def _put_back_offset(autofocus: Autofocus, start: float, tolerance: float) -> None:
    if abs(autofocus.offset_native() - start) > tolerance:
        autofocus.set_offset_native(start)


def _engage_again(autofocus: Autofocus) -> None:
    """Engage the autofocus it started engaged; no lock is a failure the put-back names (FM-49)."""
    if not autofocus.engage():
        raise RuntimeError(
            "the autofocus did not lock again, so it is off; engage it at the stand"
        )


def _put_back_exposure(camera: Camera, exposure_ms: float) -> None:
    if camera.exposure_ms() != exposure_ms:
        camera.set_exposure_ms(exposure_ms)


def _put_back_auto_shutter(shutter: Shutter, on: bool) -> None:
    if shutter.auto_shutter() != on:
        shutter.set_auto_shutter(on)
