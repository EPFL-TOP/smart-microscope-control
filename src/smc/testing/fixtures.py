"""pytest plugin: the demo, fake and hardware stands, and ``--profile`` (design §8).

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

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest

from smc.hardware.core import close_core, find_install, open_core
from smc.hardware.microscope import Microscope
from smc.hardware.profile import Profile
from smc.testing.fakes import FakeCore

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

__all__ = [
    "demo_core",
    "demo_microscope",
    "demo_microscope_dry",
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
