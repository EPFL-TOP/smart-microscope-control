"""The plugin errors (m3 §1.9): what they carry, what they say, that they pickle."""

from __future__ import annotations

import pickle

import pytest

from smc.hardware.errors import HardwareError
from smc.plugins.api import StopRequested
from smc.plugins.errors import (
    ParamsError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    RequirementsError,
    ResumeError,
    UpstreamError,
)
from smc.plugins.registry import LoadFailure
from smc.run import RunDocumentError

FAILURE = LoadFailure(
    name="broken",
    target="pkg.mod:Broken",
    distribution="pkg",
    reason="could not be imported: ImportError: no module named 'x'",
)

ERRORS: list[PluginError] = [
    PluginNotFoundError("nothing", ("a", "b")),
    PluginLoadError(FAILURE),
    ParamsError("stage-report", ("samples: Input should be a valid integer", "x")),
    RequirementsError(
        "stage-report",
        ("XYStage",),
        ("XYStage needs the 'xy_stage' role. Assign one under [roles.assign].",),
    ),
    UpstreamError("wells", "plate", "the run document has no plate"),
    ResumeError("scan", "nothing to resume"),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
def test_every_plugin_error_survives_pickle(error: PluginError) -> None:
    # Like the errors of smc.hardware: one may cross a worker process.
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error)
    assert str(copy) == str(error)
    assert vars(copy) == vars(error)


def test_a_plugin_error_is_not_a_device_fault() -> None:
    # A configuration problem is not a hardware fault, and no `except
    # HardwareError` or `except RuntimeError` of a plugin loop may take it.
    for error in ERRORS:
        assert isinstance(error, PluginError)
        assert not isinstance(error, HardwareError | RuntimeError)
    assert not issubclass(RunDocumentError, PluginError | HardwareError)


def test_a_stop_is_no_plugin_error_and_no_refusal() -> None:
    # FM-64: `except PluginError` or `except SafetyRefusedError` in a loop must
    # never swallow a stop.
    assert not issubclass(StopRequested, PluginError | HardwareError)
    assert issubclass(StopRequested, Exception)


def test_not_found_names_the_plugins_there_are() -> None:
    error = PluginNotFoundError("nothing", ("a", "b"))
    assert (error.name, error.available) == ("nothing", ("a", "b"))
    assert str(error) == (
        "no plugin named 'nothing' (available: a, b); smc run --list shows them"
    )
    assert str(PluginNotFoundError("x", ())) == (
        "no plugin named 'x' (available: none); smc run --list shows them"
    )


def test_a_load_error_says_which_plugin_and_why() -> None:
    error = PluginLoadError(FAILURE)
    assert error.failure is FAILURE
    assert str(error) == (
        "plugin broken (pkg.mod:Broken, from pkg) could not be imported: "
        "ImportError: no module named 'x'; smc run --list shows the plugins that "
        "do load"
    )


def test_params_error_lists_every_problem_and_points_at_show_params() -> None:
    error = ParamsError("stage-report", ("unknown key 'x'", "samples: too small"))
    assert error.plugin == "stage-report"
    assert error.problems == ("unknown key 'x'", "samples: too small")
    assert str(error) == (
        "parameters of stage-report: unknown key 'x'; samples: too small. "
        "smc run stage-report --show-params lists the keys"
    )


def test_requirements_error_lists_every_missing_capability_with_its_detail() -> None:
    error = RequirementsError(
        "scan",
        ("XYStage", "ZStage"),
        ("XYStage needs [roles.assign].", "ZStage needs [roles.assign]."),
    )
    assert error.missing == ("XYStage", "ZStage")
    assert str(error) == (
        "scan needs XYStage and ZStage, which this stand does not have: "
        "XYStage needs [roles.assign]. ZStage needs [roles.assign]."
    )


def test_upstream_and_resume_errors_name_the_plugin_and_say_what_to_do() -> None:
    upstream = UpstreamError("wells", "plate", "register ended failed; run it again")
    assert (upstream.plugin, upstream.slot, upstream.reason) == (
        "wells",
        "plate",
        "register ended failed; run it again",
    )
    assert str(upstream) == (
        "wells reads plate, but register ended failed; run it again"
    )
    resume = ResumeError("scan", "nothing to resume")
    assert (resume.plugin, resume.reason) == ("scan", "nothing to resume")
    assert str(resume) == "cannot resume scan: nothing to resume"
