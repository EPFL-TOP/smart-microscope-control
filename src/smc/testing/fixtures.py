"""pytest plugin: the demo, fake and hardware stands, and ``--profile`` (design §8).

``demo_microscope_with_sample`` is the demo stand with a camera whose frames
follow the stage (``smc.testing.synthetic``); the others give the stand as it
is.

Load it from a top-level ``conftest.py`` with
``pytest_plugins = ["smc.testing.fixtures"]``; a plugin's own repository
does the same. It imports pytest, a dev dependency, so ``smc.testing``
does not import it.

Two kinds of test use these fixtures (ADR-0005):

* **simulator tests** run on Micro-Manager's demo devices, everywhere,
  CI included. Without the adapters they *skip* locally, but *fail* when
  ``SMC_REQUIRE_MM=1`` (CI), so a broken install cannot hide behind a
  green run.
* **hardware tests** carry ``@pytest.mark.hardware`` and run only at a
  microscope: ``pytest -m hardware --profile <name>``.

``hardware_microscope`` refuses a test that is not marked ``hardware``
*before* it opens anything, so a default run, which deselects that marker,
can never reach a stand through a test that forgot it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from smc.hardware.capabilities import Camera
from smc.hardware.core import close_core, find_install, open_core
from smc.hardware.microscope import Microscope
from smc.hardware.profile import Profile
from smc.testing.fakes import FakeCore
from smc.testing.synthetic import PlateSample, SampleCamera

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = [
    "demo_core",
    "demo_microscope",
    "demo_microscope_dry",
    "demo_microscope_with_sample",
    "fake_core",
    "fake_microscope",
    "hardware_microscope",
    "mm_available",
    "pytest_addoption",
]

#: Why a hardware test was skipped: it is the line a user sees at the stand.
NO_PROFILE = (
    "hardware tests need --profile <name>; run them only at the microscope (ADR-0005)"
)
#: Why a test that reaches for the stand without the marker fails.
NOT_MARKED = "hardware_microscope is only for tests marked @pytest.mark.hardware"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--profile``: the stand hardware tests open, by name or path."""
    parser.addoption(
        "--profile",
        default=None,
        help="Microscope profile for hardware tests, a name or a path "
        "(e.g. nikon-ti2); `demo` rehearses on the simulator.",
    )


@pytest.fixture(scope="session")
def mm_available() -> bool:
    """Whether the Micro-Manager device adapters are installed.

    Fails the session's first user under ``SMC_REQUIRE_MM=1`` when they are
    not, so CI cannot skip its way to green.
    """
    available = find_install() is not None
    if not available and os.environ.get("SMC_REQUIRE_MM") == "1":
        pytest.fail(
            "SMC_REQUIRE_MM=1 but Micro-Manager adapters are missing — "
            "run `mmcore install --test-adapters`."
        )
    return available


def _need_mm(available: bool) -> None:
    if not available:
        pytest.skip("Micro-Manager demo adapters not installed")


@pytest.fixture
def demo_core(mm_available: bool) -> Iterator[CMMCorePlus]:
    """A fresh core loaded with the demo configuration, released afterwards."""
    _need_mm(mm_available)
    core = open_core(None)
    try:
        yield core
    finally:
        close_core(core)


def _opened(profile: Profile, *, dry_run: bool) -> Iterator[Microscope]:
    microscope = Microscope.open(profile, dry_run=dry_run)
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def demo_microscope(mm_available: bool) -> Iterator[Microscope]:
    """The demo stand, opened as a tool opens it, closed afterwards.

    ``Profile.demo()``, never ``"demo"``: ``Profile.load("demo")`` would pick
    up whatever ``demo.toml`` the working directory has.
    """
    _need_mm(mm_available)
    yield from _opened(Profile.demo(), dry_run=False)


@pytest.fixture
def demo_microscope_dry(mm_available: bool) -> Iterator[Microscope]:
    """The demo stand in dry-run: reads reach the devices, mutations are skipped."""
    _need_mm(mm_available)
    yield from _opened(Profile.demo(), dry_run=True)


class SampleMicroscopeFactory(Protocol):
    """What ``demo_microscope_with_sample`` gives a test: a function that returns the stand."""

    def __call__(
        self,
        sample: PlateSample | None = None,
        *,
        pixel_size_um: float | None = None,
        shape: tuple[int, int] | None = None,
        camera_rotation_deg: float = 0.0,
        mirrored: bool = False,
    ) -> Microscope: ...


@pytest.fixture
def demo_microscope_with_sample(
    demo_microscope: Microscope,
) -> SampleMicroscopeFactory:
    """A factory that puts a ``SampleCamera`` on the demo stand and returns the stand.

    The demo camera ignores the stage, so a test that needs frames that follow
    it asks for this instead. Call it once per test, with the sample to look
    at; the default is ``PlateSample()``, which puts A1's centre at the demo
    stage's origin. The frame shape and the pixel size are those of the camera
    the stand has *now*, read before it is replaced, unless the call gives its
    own. Calling it again replaces the sample camera.

    The sample camera bypasses the ``Executor`` (see ``SampleCamera``). On
    ``fake_microscope``, build the ``SampleCamera`` yourself.

    The returned function raises ``ValueError`` when the stand's camera reports
    no pixel size and the call gives none: a pixel size is measured or given,
    never guessed. Nothing is replaced in that case.
    """

    def make(
        sample: PlateSample | None = None,
        *,
        pixel_size_um: float | None = None,
        shape: tuple[int, int] | None = None,
        camera_rotation_deg: float = 0.0,
        mirrored: bool = False,
    ) -> Microscope:
        current = demo_microscope.require(Camera)
        if pixel_size_um is None:
            pixel_size_um = current.pixel_size_um()
            if not (math.isfinite(pixel_size_um) and pixel_size_um > 0.0):
                raise ValueError(
                    f"the stand's camera reports no pixel size ({pixel_size_um!r}), "
                    "so the sample camera cannot take it: give pixel_size_um=<um per pixel>"
                )
        camera = SampleCamera(
            demo_microscope,
            sample if sample is not None else PlateSample(),
            pixel_size_um=pixel_size_um,
            shape=current.image_shape() if shape is None else shape,
            camera_rotation_deg=camera_rotation_deg,
            mirrored=mirrored,
        )
        demo_microscope.override(Camera, camera)
        return demo_microscope

    return make


@pytest.fixture
def fake_core() -> FakeCore:
    """``FakeCore.demo_like()``: the demo's devices, without Micro-Manager."""
    return FakeCore.demo_like()


@pytest.fixture
def fake_microscope(fake_core: FakeCore) -> Iterator[Microscope]:
    """The facade over ``fake_core`` with the demo profile, closed afterwards.

    Building it logs ``setTimeoutMs`` on the fake; clear ``fake_core.log``
    before asserting on it.
    """
    # The facade types its core as CMMCorePlus; the fake is structural.
    microscope = Microscope.from_core(cast("CMMCorePlus", fake_core), Profile.demo())
    try:
        yield microscope
    finally:
        microscope.close()


@pytest.fixture
def hardware_microscope(request: pytest.FixtureRequest) -> Iterator[Microscope]:
    """The stand ``--profile`` names, opened for a hardware test, closed afterwards.

    Checked in this order, before anything is opened: the test must be
    marked ``@pytest.mark.hardware`` (else it fails), and ``--profile`` must
    be given (else it skips). ``--profile demo`` rehearses on the simulator.
    """
    if request.node.get_closest_marker("hardware") is None:
        pytest.fail(NOT_MARKED)
    profile = cast("str | None", request.config.getoption("--profile"))
    if not profile:
        pytest.skip(NO_PROFILE)
    yield from _opened(Profile.load(profile), dry_run=False)
