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

#: Makes any open of a stand raise, so a test can tell "refused before
#: opening" from "opened, then refused" on every OS: on Windows the inner run
#: still finds the demo adapters (see the MICROMANAGER_PATH note below).
OPENING = "the stand was opened"
NO_OPEN = f"""
import smc.hardware.microscope

def _open(cls, *args, **kwargs):
    raise RuntimeError({OPENING!r})

smc.hardware.microscope.Microscope.open = classmethod(_open)
"""


#: Bounds each inner run: pytester waits for ever by default, so an inner
#: run stuck on a modal error dialog (a DLL that fails to load on Windows)
#: would hang the CI job with no diagnosis.
INNER_TIMEOUT_S = 120.0


def _run(
    pytester: pytest.Pytester, source: str, *args: str, opens: bool = True
) -> pytest.RunResult:
    pytester.makeini(INI)
    pytester.makepyfile(source)
    if not opens:
        pytester.makeconftest(NO_OPEN)
    return pytester.runpytest_subprocess(
        "-p", "smc.testing.fixtures", "-rsE", *args, timeout=INNER_TIMEOUT_S
    )


def test_hardware_microscope_is_skipped_without_profile(
    pytester: pytest.Pytester,
) -> None:
    result = _run(pytester, MARKED, "-m", "hardware")
    result.assert_outcomes(skipped=1)
    assert NO_PROFILE in result.stdout.str()


def test_hardware_microscope_fails_a_test_not_marked_hardware(
    pytester: pytest.Pytester,
) -> None:
    # The marker guard comes before the open: with --profile given, a guard
    # placed after it would reach OPENING first.
    result = _run(pytester, UNMARKED, "--profile", "demo", opens=False)
    result.assert_outcomes(errors=1)
    assert NOT_MARKED in result.stdout.str()
    assert OPENING not in result.stdout.str()


def test_hardware_microscope_fails_an_unmarked_test_even_without_profile(
    pytester: pytest.Pytester,
) -> None:
    # The marker guard comes before the --profile skip: the other way round,
    # a forgotten marker would hide behind an innocent-looking skip.
    result = _run(pytester, UNMARKED, opens=False)
    result.assert_outcomes(errors=1)
    assert NOT_MARKED in result.stdout.str()


@pytest.mark.demo
def test_hardware_microscope_opens_the_named_profile(
    pytester: pytest.Pytester, mm_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mm_available:
        pytest.skip("Micro-Manager demo adapters not installed")
    # pytester points HOME at a temporary directory, where pymmcore-plus no
    # longer finds its per-user install on macOS and Linux (Windows asks the
    # OS for %LOCALAPPDATA% instead). Name the one this session found, so
    # every OS opens the same install.
    monkeypatch.setenv("MICROMANAGER_PATH", str(find_install()))
    result = _run(pytester, MARKED, "--profile", "demo", "-m", "hardware")
    result.assert_outcomes(passed=1)


def test_smc_testing_imports_without_pytest() -> None:
    # pytest is a dev dependency; a plugin's users import the fake without it.
    code = (
        "import smc.testing, sys; "
        "from smc.testing import Blob, PlateSample, SampleCamera; "
        "assert 'pytest' not in sys.modules, 'pytest'"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # A Windows child writes a pipe in its ANSI code page: a strict decode
        # would hide the child's error behind a UnicodeDecodeError.
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
