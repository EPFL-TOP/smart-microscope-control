"""The pytest plugin's skip and refusal rules, run in a separate pytest (#9).

``hardware_microscope`` is the one door from the test suite to a stand, so
its rules are tested as a user meets them: an inner pytest run in a
subprocess (``runpytest_subprocess``), which is isolated from this session,
where the plugin is already registered.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from smc.hardware.core import find_install
from smc.testing.fixtures import NO_PROFILE, NOT_MARKED

INI = """
[pytest]
markers =
    hardware: needs a real microscope
"""

MARKED = """
import pytest

@pytest.mark.hardware
def test_at_the_stand(hardware_microscope):
    assert hardware_microscope.profile.microscope.name == "demo"
"""

UNMARKED = """
def test_forgot_the_marker(hardware_microscope):
    pass
"""


def _run(pytester: pytest.Pytester, source: str, *args: str) -> pytest.RunResult:
    pytester.makeini(INI)
    pytester.makepyfile(source)
    return pytester.runpytest_subprocess("-p", "smc.testing.fixtures", "-rsE", *args)


def test_hardware_microscope_is_skipped_without_profile(
    pytester: pytest.Pytester,
) -> None:
    result = _run(pytester, MARKED, "-m", "hardware")
    result.assert_outcomes(skipped=1)
    assert NO_PROFILE in result.stdout.str()


def test_hardware_microscope_fails_a_test_not_marked_hardware(
    pytester: pytest.Pytester,
) -> None:
    # The guard comes before the open, so this needs no Micro-Manager: with
    # the order reversed, the run would try to open the stand, and the
    # error would not be the guard's.
    result = _run(pytester, UNMARKED, "--profile", "demo")
    result.assert_outcomes(errors=1)
    assert NOT_MARKED in result.stdout.str()


@pytest.mark.demo
def test_hardware_microscope_opens_the_named_profile(
    pytester: pytest.Pytester, mm_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    # pytester points HOME at a temporary directory, where pymmcore-plus no
    # longer finds its per-user install; name the one this session found.
    monkeypatch.setenv("MICROMANAGER_PATH", str(find_install()))
    result = _run(pytester, MARKED, "--profile", "demo", "-m", "hardware")
    result.assert_outcomes(passed=1)


def test_smc_testing_imports_without_pytest() -> None:
    # pytest is a dev dependency; a plugin's users import the fake without it.
    code = "import smc.testing, sys; assert 'pytest' not in sys.modules, 'pytest'"
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
