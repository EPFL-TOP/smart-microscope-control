"""``Microscope.offline``: a facade with no core and no device (algorithms §4.2, #566).

An offline facade is what a plugin that never drives the stand runs on, on a
run folder copied off the stand. Nothing here opens a core: a test that did
would fail on ``open_core`` (FM-40), and the demo adapters are not needed.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import sys
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import smc.hardware.microscope as microscope_mod
from smc.hardware import Microscope, MicroscopeState
from smc.hardware.capabilities import XYStage
from smc.hardware.errors import CapabilityMissingError, HardwareError
from smc.hardware.microscope import CAPABILITY_ROLES
from smc.hardware.profile import Profile
from smc.hardware.roles import Role
from smc.testing.fakes import FakeCore

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_core_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here runs with the ways to open or free a core made to fail."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("an offline microscope touched a core")

    monkeypatch.setattr(microscope_mod, "open_core", fail)
    monkeypatch.setattr(microscope_mod, "close_core", fail)
    monkeypatch.setattr(microscope_mod, "detach_callbacks", fail)


def test_an_offline_microscope_has_no_capability() -> None:
    microscope = Microscope.offline(Profile.demo())

    for capability in CAPABILITY_ROLES:
        assert microscope.has(capability) is False, capability.__name__
        assert microscope.get(capability) is None, capability.__name__
    assert microscope.available() == []


def test_require_on_an_offline_microscope_says_no_stand_is_open() -> None:
    microscope = Microscope.offline(Profile.demo())

    # Channels has a branch of its own on a stand: offline it must not reach it,
    # which would ask the core for its config groups.
    for capability in CAPABILITY_ROLES:
        with pytest.raises(CapabilityMissingError) as raised:
            microscope.require(capability)
        assert "no stand is open" in str(raised.value), capability.__name__
        assert raised.value.capability == capability.__name__
    with pytest.raises(CapabilityMissingError, match="XYStage is not available"):
        microscope.require(XYStage)


def test_an_offline_microscope_answers_every_other_member_without_a_core() -> None:
    profile = Profile.demo()
    with Microscope.offline(profile) as microscope:
        assert microscope.is_offline is True
        assert microscope.profile is profile
        assert microscope.dry_run is False
        assert microscope.devices == []
        assert microscope.roles.assigned == {}
        assert microscope.roles.warnings == []
        assert microscope.roles.missing() == list(Role)
        assert microscope.unload_error is None
        assert microscope.describe() == "offline (no stand is open)"
        # Nothing is read: every field is the unset one.
        assert microscope.state() == MicroscopeState()
        with pytest.raises(HardwareError, match="no stand is open"):
            _ = microscope.core
        microscope.resume()
        microscope.stop()
        assert microscope.state().halted is True
        microscope.resume()
        assert microscope.state().halted is False
    # The `with` block closed it, and closing again is harmless.
    microscope.close()
    microscope.close()
    assert microscope.unload_error is None


def test_stop_on_an_offline_microscope_raises_nothing() -> None:
    # An operator's Ctrl-C handler calls stop(): with no stage there is nothing
    # to stop, and it must not raise for it.
    microscope = Microscope.offline(Profile.demo())

    microscope.stop()
    microscope.stop()
    microscope.close()


def test_override_on_an_offline_microscope_is_refused() -> None:
    microscope = Microscope.offline(Profile.demo())

    with pytest.raises(HardwareError, match="there is nothing to override"):
        microscope.override(XYStage, object())
    assert microscope.has(XYStage) is False


def test_a_microscope_built_from_a_core_is_not_offline_and_keeps_its_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), Profile.demo())
    try:
        assert microscope.is_offline is False
        assert microscope.core is core
        assert microscope.has(XYStage) is True
        assert microscope.devices != []
        assert microscope.describe() != "offline (no stand is open)"
    finally:
        # close() unloads the fake's devices; detach_callbacks is the failure of
        # the module's fixture, and the fake core has no relay to detach.
        monkeypatch.setattr(microscope_mod, "detach_callbacks", lambda core: None)
        microscope.close()


def test_creating_an_offline_microscope_never_opens_a_core(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The autouse fixture makes open_core, close_core and detach_callbacks fail.
    caplog.set_level(logging.DEBUG, logger="smc.hardware.microscope")

    with Microscope.offline(Profile.demo()) as microscope:
        assert microscope.is_offline
    # No role warning, no Z jog default warning: there is no stand to warn about.
    assert [record.getMessage() for record in caplog.records] == []


def test_an_offline_microscope_holds_no_reference_cycle(
    no_cyclic_gc: None,
) -> None:
    # FM-92: the facade is freed by reference counting alone, as at a CLI
    # command's end; `core` is a property and nothing built here refers back.
    microscope = Microscope.offline(Profile.demo())
    microscope.state()
    microscope.describe()
    microscope.close()
    alive = weakref.ref(microscope)
    del microscope
    assert alive() is None
    assert gc.garbage == []


def test_creating_an_offline_microscope_does_not_import_pymmcore_plus() -> None:
    code = (
        "import sys\n"
        "from smc.hardware import Microscope\n"
        "from smc.hardware.profile import Profile\n"
        "microscope = Microscope.offline(Profile.load('nikon-ti2'))\n"
        "microscope.state()\n"
        "microscope.close()\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0].startswith('pymmcore'))\n"
        "print('LOADED', loaded)\n"
    )

    done = subprocess.run(
        # -X utf8: stdout and stderr are read as UTF-8, whatever the console's code page.
        [sys.executable, "-I", "-X", "utf8", "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        cwd=ROOT,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"
