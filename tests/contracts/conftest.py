"""Contract fixtures: one stand per backend, its envelope, and the put-back (design §8).

Every contract runs on three backends: ``fake`` (``FakeCore``), ``demo``
(Micro-Manager's demo devices) and ``hardware`` (the stand ``--profile``
names; deselected unless ``-m hardware``). The tests never import a
backend class: they only go through ``Microscope``.

**What a contract may move.** ``envelope`` records where the stand started
and replaces the profile's ``[safety]`` with limits around that start: XY
± 100 µm, Z ± 3 µm, a 50 µm jog limit. Every target a contract *sends* is
within 60 µm in XY, and in Z at most 2 µm and only below the start, which
is away from the sample on an inverted stand. Every target outside the
envelope is one the layer must refuse, so none of them is ever sent. The
``microscope`` fixture then puts the stand back where it started.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

import pytest

from smc.hardware import Microscope
from smc.hardware.capabilities import (
    XY,
    Camera,
    Properties,
    Shutter,
    XYStage,
    ZStage,
)
from smc.hardware.profile import Profile, SafetySection

T = TypeVar("T")

#: The fixture that opens each backend's stand; it owns the core and closes it.
_STANDS = {
    "fake": "fake_microscope",
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
    max_jog_um: float = 50.0
    xy_span_um: float = 100.0
    z_span_um: float = 3.0
    tolerance_um: float = 0.5

    def profile(self, base: Profile) -> Profile:
        """``base`` with soft limits around the start and the contract's jog limit.

        Every other key of ``base`` is kept, ``[safety]``'s included (such as a
        later ``max_z_jog_um``), and so is ``source``.

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
        safety = SafetySection.model_validate(
            {
                **base.safety.model_dump(),
                "max_jog_um": self.max_jog_um,
                "xy_soft_limits_um": xy_limits,
                "z_soft_limits_um": z_limits,
            }
        )
        return base.model_copy(update={"safety": safety})


def _require_inside(
    axis: str, window: tuple[float, float], limit: tuple[float, float]
) -> None:
    """Fail the test unless the contract's ``window`` lies within the profile's ``limit``."""
    if window[0] < limit[0] or window[1] > limit[1]:
        pytest.fail(
            f"the stand starts too close to its {axis} soft limits "
            f"[{limit[0]:g}, {limit[1]:g}] µm: the contracts need "
            f"[{window[0]:g}, {window[1]:g}] µm. Move it towards the middle of "
            f"its travel and run them again"
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
    return Envelope(
        xy=None if xy is None else xy.position_um(),
        z_um=None if z is None else z.position_um(),
        exposure_ms=None if camera is None else camera.exposure_ms(),
        shutter_open=None if shutter is None else shutter.is_open(),
        auto_shutter=None if shutter is None else shutter.auto_shutter(),
    )


@pytest.fixture
def microscope(stand: Microscope, envelope: Envelope) -> Iterator[Microscope]:
    """A second facade over ``stand``'s core, under the envelope's ``[safety]``.

    Never closed: ``stand`` owns the core. Its teardown puts the stand back.
    """
    facade = Microscope.from_core(stand.core, envelope.profile(stand.profile))
    yield facade
    _put_back(facade, envelope)


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
def dry_xy(dry_microscope: Microscope, backend: str) -> XYStage:
    """The XY stage in dry-run."""
    return _capability(dry_microscope, backend, XYStage)


@pytest.fixture
def dry_z(dry_microscope: Microscope, backend: str) -> ZStage:
    """The Z drive in dry-run."""
    return _capability(dry_microscope, backend, ZStage)


@pytest.fixture
def put_back() -> Callable[[Microscope, Envelope], None]:
    """The put-back ``microscope`` runs at teardown, for the tests of it."""
    return _put_back


def _put_back(facade: Microscope, envelope: Envelope) -> None:
    """Return the stand to ``envelope``, or stop it if a motion is left.

    A halted facade, or one with a motion still registered, is stopped
    instead: the stand's own ``close()`` cannot see this facade's motions.
    The light is still cut if it started off, since closing a shutter is
    never refused (§13). Otherwise every item that differs is put back,
    light off first and on last.

    Every step is attempted even if an earlier one failed, and one error
    then names every step that failed: at a stand, the operator must learn
    each item that was not put back, not only the first.
    """
    shutter = facade.get(Shutter)
    state = facade.state()
    steps: list[tuple[str, Callable[[], None]]] = []
    if state.halted or state.moving:
        steps.append(("stop", facade.stop))
        if shutter is not None and envelope.shutter_open is False:
            steps.append(("shutter close", lambda: _put_back_shutter(shutter, False)))
        _run_all(steps)
        return

    tolerance_um = envelope.tolerance_um
    if shutter is not None and envelope.shutter_open is False:
        steps.append(("shutter close", lambda: _put_back_shutter(shutter, False)))
    xy = facade.get(XYStage)
    if xy is not None and (start := envelope.xy) is not None:
        steps.append(("xy", lambda: _put_back_xy(xy, start, tolerance_um)))
    z = facade.get(ZStage)
    if z is not None and (z0 := envelope.z_um) is not None:
        steps.append(("z", lambda: _put_back_z(z, z0, tolerance_um)))
    camera = facade.get(Camera)
    if camera is not None and (ms := envelope.exposure_ms) is not None:
        steps.append(("exposure", lambda: _put_back_exposure(camera, ms)))
    if shutter is not None and (auto := envelope.auto_shutter) is not None:
        steps.append(("auto-shutter", lambda: _put_back_auto_shutter(shutter, auto)))
    if shutter is not None and envelope.shutter_open is True:
        steps.append(("shutter open", lambda: _put_back_shutter(shutter, True)))
    _run_all(steps)


def _run_all(steps: list[tuple[str, Callable[[], None]]]) -> None:
    """Run every step; then raise one error naming each that failed, from the first."""
    failed: list[tuple[str, Exception]] = []
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            failed.append((name, exc))
    if failed:
        listed = "; ".join(f"{name}: {exc!r}" for name, exc in failed)
        raise RuntimeError(
            f"the stand was not put back where it started ({listed}); check it "
            f"before the next run"
        ) from failed[0][1]


def _put_back_shutter(shutter: Shutter, open_: bool) -> None:
    if shutter.is_open() != open_:
        shutter.set_open(open_)


def _put_back_xy(xy: XYStage, start: XY, tolerance_um: float) -> None:
    here = xy.position_um()
    if max(abs(here.x_um - start.x_um), abs(here.y_um - start.y_um)) > tolerance_um:
        xy.move_to_um(start.x_um, start.y_um)


def _put_back_z(z: ZStage, z0_um: float, tolerance_um: float) -> None:
    if abs(z.position_um() - z0_um) > tolerance_um:
        z.move_to_um(z0_um)


def _put_back_exposure(camera: Camera, exposure_ms: float) -> None:
    if camera.exposure_ms() != exposure_ms:
        camera.set_exposure_ms(exposure_ms)


def _put_back_auto_shutter(shutter: Shutter, on: bool) -> None:
    if shutter.auto_shutter() != on:
        shutter.set_auto_shutter(on)
