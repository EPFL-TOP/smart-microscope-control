"""The plugin registry (m3 §1.5): discovery is a list of findings, never a crash.

Test plugins are classes of this module, loaded through
``EntryPoint(value=f"{__name__}:<Class>")``: a module already in ``sys.modules``
needs no installed distribution (m3 §1.1). Nothing here imports a vendor adapter
or opens a stand (FM-40).
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

import smc
from smc.hardware.capabilities import Camera, XYStage, ZStage
from smc.plugins.api import Plugin, PluginParams, PluginResult, RunContext
from smc.plugins.errors import PluginLoadError, PluginNotFoundError
from smc.plugins.registry import (
    GROUP,
    LoadFailure,
    PluginEntry,
    Registry,
    discover,
    load,
)

THIS = __name__


class GoodParams(PluginParams):
    pass


class GoodResult(PluginResult):
    pass


class Good:
    name = "good"
    summary = "Does nothing, correctly."
    Params = GoodParams
    Result = GoodResult
    requires: tuple[type, ...] = (XYStage,)
    provides = "good"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope: Any, params: GoodParams, ctx: RunContext) -> GoodResult:
        return GoodResult()


class Other(Good):
    name = "other"
    provides = "other"


class Exploding:
    def __init__(self) -> None:
        raise RuntimeError("boom")


class NoRun:
    name = "no-run"
    summary = "x"
    Params = GoodParams
    Result = GoodResult
    requires: tuple[type, ...] = ()
    provides = "no-run"
    reads: tuple[str, ...] = ()
    # no reads_optional, no run


class Mismatch(Good):
    name = "something-else"


class Shouting(Good):
    name = "Good_Plugin"


class NotKebabSlot(Good):
    provides = "Plate"


class StringReads(Good):
    reads = "plate"  # type: ignore[assignment]


class BadReadSlot(Good):
    reads = ("plate", "Bad Slot")


class BadOptionalSlot(Good):
    reads_optional = ("timings", "9lives")


class Overlap(Good):
    reads = ("plate", "timings")
    reads_optional = ("timings", "calibration")


class ReadsItsOwnSlot(Good):
    reads = ("good",)


class OptionalReadsItsOwnSlot(Good):
    reads_optional = ("good",)


class WrongParams(Good):
    Params = dict  # type: ignore[assignment]


class WrongResult(Good):
    Result = int  # type: ignore[assignment]


class NotACapability(Good):
    requires = (XYStage, int)


class RequiresText(Good):
    requires = ("XYStage",)  # type: ignore[assignment]


class RequiresEverything(Good):
    requires = (XYStage, ZStage, Camera)


class RequiresList(Good):
    requires: ClassVar[list[type]] = [XYStage]  # type: ignore[assignment]


class NameNotText(Good):
    name = 3  # type: ignore[assignment]


class NoSummary(Good):
    summary = None  # type: ignore[assignment]


class RaisingSummary(Good):
    @property
    def summary(self) -> str:  # type: ignore[override]
        raise RuntimeError("config file missing")


class BadNameAndBadModel(Good):
    name = "Bad Name"
    Params = dict  # type: ignore[assignment]


NOT_A_CLASS = "just a string"


def _ep(name: str, target: str, *, module: str = THIS) -> EntryPoint:
    return EntryPoint(name=name, value=f"{module}:{target}", group=GROUP)


def _failure(registry: Registry, name: str) -> LoadFailure:
    (found,) = [f for f in registry.failures if f.name == name]
    return found


def _reason(cls_name: str, *, name: str | None = None) -> str:
    registry = discover([_ep(name or "good", cls_name)])
    assert registry.plugins == {}
    return registry.failures[0].reason


def test_stage_report_is_discovered_through_its_entry_point() -> None:
    # The installed metadata, no injected list: this fails until pyproject.toml's
    # entry point is installed (a fresh install on CI; `worktree.py setup` here).
    registry = discover()
    assert registry.failures == ()
    entry = registry.plugins["stage-report"]
    assert entry.target == "smc.plugins.stage_report:StageReport"
    assert entry.distribution == "smart-microscope-control"
    assert entry.version == smc.__version__
    assert entry.plugin.requires == (XYStage,)
    assert load("stage-report").plugin.name == "stage-report"


def test_a_good_plugin_is_loaded_with_where_it_came_from() -> None:
    registry = discover([_ep("good", "Good")])
    assert registry.failures == ()
    entry = registry.plugins["good"]
    assert isinstance(entry, PluginEntry)
    assert isinstance(entry.plugin, Good)
    assert isinstance(entry.plugin, Plugin)
    assert entry.target == f"{THIS}:Good"
    assert (entry.distribution, entry.version) == (None, None)


def test_a_plugin_that_fails_to_import_is_a_finding_and_the_others_load() -> None:
    registry = discover(
        [
            _ep("good", "Good"),
            _ep("ghost", "Thing", module="no_such_module_for_smc_tests"),
        ]
    )
    assert list(registry.plugins) == ["good"]
    assert registry.failures == (
        LoadFailure(
            name="ghost",
            target="no_such_module_for_smc_tests:Thing",
            distribution=None,
            reason="could not be imported: ModuleNotFoundError: No module named "
            "'no_such_module_for_smc_tests'",
        ),
    )


def test_an_attribute_that_does_not_exist_is_an_import_finding() -> None:
    assert _reason("Nothing") == (
        f"could not be imported: AttributeError: module {THIS!r} has no attribute "
        f"'Nothing'"
    )


def test_a_name_that_differs_from_the_entry_point_is_refused() -> None:
    assert _reason("Mismatch") == (
        "is named 'something-else', but its entry point is named 'good'; they must "
        "be equal"
    )


def test_two_targets_for_one_name_are_both_refused() -> None:
    registry = discover([_ep("good", "Good"), _ep("good", "Other")])
    assert registry.plugins == {}
    reason = (
        f"registered twice: {THIS}:Good (no distribution), {THIS}:Other "
        f"(no distribution)"
    )
    assert [(f.target, f.reason) for f in registry.failures] == [
        (f"{THIS}:Good", reason),
        (f"{THIS}:Other", reason),
    ]


def test_an_identical_entry_listed_twice_loads_once() -> None:
    # An editable install can list one distribution twice.
    registry = discover([_ep("good", "Good"), _ep("good", "Good")])
    assert registry.failures == ()
    assert list(registry.plugins) == ["good"]


def test_plugins_are_ordered_by_name_and_failures_by_name_then_target() -> None:
    registry = discover(
        [
            _ep("zeta", "Good"),
            _ep("other", "Other"),
            _ep("zz", "Nothing"),
            _ep("aa", "Exploding"),
            _ep("good", "Good"),
        ]
    )
    # "zeta" names Good, whose name is "good": refused, like the others.
    assert list(registry.plugins) == ["good", "other"]
    assert [f.name for f in registry.failures] == ["aa", "zeta", "zz"]


@pytest.fixture
def marker_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EntryPoint, Path]:
    """An entry whose module leaves a file behind when it is imported.

    A failed import leaves nothing in ``sys.modules``, so "was it imported?"
    needs a trace of its own (FM-43).
    """
    marker = tmp_path / "imported.marker"
    module = tmp_path / "smc_marker_plugin_module.py"
    module.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "class Marker:\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "smc_marker_plugin_module", raising=False)
    return _ep("marker", "Marker", module="smc_marker_plugin_module"), marker


def test_load_imports_only_the_named_entry(
    marker_entry: tuple[EntryPoint, Path],
) -> None:
    marker_ep, marker = marker_entry
    entry = load("good", [_ep("good", "Good"), marker_ep])
    assert isinstance(entry.plugin, Good)
    assert not marker.exists()
    # Positive control: `discover` does import it, and the marker shows it.
    discover([_ep("good", "Good"), marker_ep])
    assert marker.read_text(encoding="utf-8") == "imported"


def test_load_of_an_unknown_name_lists_the_available_ones(
    marker_entry: tuple[EntryPoint, Path],
) -> None:
    marker_ep, marker = marker_entry
    with pytest.raises(PluginNotFoundError) as raised:
        load("nothing", [_ep("good", "Good"), marker_ep])
    assert raised.value.name == "nothing"
    assert raised.value.available == ("good", "marker")  # read, never imported
    assert not marker.exists()
    assert str(raised.value) == (
        "no plugin named 'nothing' (available: good, marker); smc run --list shows them"
    )


def test_load_of_a_plugin_that_fails_raises_a_load_error_with_the_finding() -> None:
    with pytest.raises(PluginLoadError) as raised:
        load("good", [_ep("good", "Mismatch")])
    assert raised.value.failure == LoadFailure(
        name="good",
        target=f"{THIS}:Mismatch",
        distribution=None,
        reason="is named 'something-else', but its entry point is named 'good'; "
        "they must be equal",
    )


def test_load_of_two_targets_for_one_name_raises_for_the_first() -> None:
    with pytest.raises(PluginLoadError) as raised:
        load("good", [_ep("good", "Other"), _ep("good", "Good")])
    assert raised.value.failure.target == f"{THIS}:Good"
    assert raised.value.failure.reason.startswith("registered twice: ")


# --- what is not a plugin ----------------------------------------------------


def test_a_target_that_is_not_a_class_is_named_by_its_type() -> None:
    assert _reason("NOT_A_CLASS") == "names a str, not a class"


def test_a_class_that_cannot_be_created_says_why() -> None:
    assert _reason("Exploding") == "could not be created: RuntimeError: boom"


def test_a_missing_member_is_named() -> None:
    assert _reason("NoRun", name="no-run") == "lacks reads_optional, run"


# --- model checks ------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "reason"),
    [
        pytest.param("NameNotText", "name must be a string, got 3", id="name-not-text"),
        pytest.param(
            "NotKebabSlot",
            "provides 'Plate', which is not a slot name (kebab-case, as a plugin name)",
            id="provides-not-kebab",
        ),
        pytest.param(
            "StringReads",
            "reads must be a tuple of slot names, got 'plate'",
            id="reads-a-string",
        ),
        pytest.param(
            "BadReadSlot",
            "reads 'Bad Slot', which is not a slot name (kebab-case, as a plugin name)",
            id="reads-not-kebab",
        ),
        pytest.param(
            "BadOptionalSlot",
            "reads_optional '9lives', which is not a slot name (kebab-case, as a "
            "plugin name)",
            id="reads-optional-not-kebab",
        ),
        pytest.param(
            "Overlap",
            "lists timings in both reads and reads_optional",
            id="reads-and-optional-overlap",
        ),
        pytest.param(
            "ReadsItsOwnSlot",
            "reads holds 'good', the slot it provides itself",
            id="reads-holding-provides",
        ),
        pytest.param(
            "OptionalReadsItsOwnSlot",
            "reads_optional holds 'good', the slot it provides itself",
            id="reads-optional-holding-provides",
        ),
    ],
)
def test_a_plugin_that_breaks_a_naming_rule_is_refused_with_the_rule(
    cls: str, reason: str
) -> None:
    assert _reason(cls) == reason


def test_a_name_that_is_not_kebab_case_is_refused_even_when_the_entry_agrees() -> None:
    assert _reason("Shouting", name="Good_Plugin") == (
        "is named 'Good_Plugin', which is not kebab-case (lowercase letters and "
        "digits, single hyphens between them, starting with a letter)"
    )


def test_a_summary_that_is_not_text_is_refused() -> None:
    # `smc run --list` prints it for every plugin: one that is not text would end
    # the listing of the others.
    assert _reason("NoSummary") == "summary must be a string, got None"


def test_a_member_that_raises_when_it_is_read_is_a_finding_not_a_crash() -> None:
    registry = discover([_ep("good", "Good"), _ep("raising", "RaisingSummary")])
    assert list(registry.plugins) == ["good"]
    assert [(f.name, f.reason) for f in registry.failures] == [
        ("raising", "could not be inspected: RuntimeError: config file missing")
    ]


def test_the_naming_checks_come_before_the_model_checks() -> None:
    # Both are false: the first of §1.5's order is the finding (FM-43).
    assert _reason("BadNameAndBadModel", name="Bad Name").startswith(
        "is named 'Bad Name', which is not kebab-case"
    )


def test_params_not_derived_from_plugin_params_is_refused() -> None:
    assert _reason("WrongParams") == "Params dict is not a subclass of PluginParams"


def test_result_not_derived_from_plugin_result_is_refused() -> None:
    assert _reason("WrongResult") == "Result int is not a subclass of PluginResult"


def test_a_requirement_that_is_not_a_capability_is_refused() -> None:
    assert _reason("NotACapability") == "requires int, which is not a capability"


def test_a_requirement_that_is_a_name_not_a_class_is_refused() -> None:
    assert _reason("RequiresText") == "requires 'XYStage', which is not a capability"


def test_requires_that_is_not_a_tuple_is_refused() -> None:
    assert _reason("RequiresList") == (
        "requires must be a tuple of capability classes, got [XYStage]"
    )


def test_every_capability_of_the_facade_may_be_required() -> None:
    registry = discover([_ep("good", "RequiresEverything")])
    assert registry.failures == ()
    assert registry.plugins["good"].plugin.requires == (XYStage, ZStage, Camera)


# --- interrupts pass through -------------------------------------------------


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_keyboard_interrupt_during_import_passes_through(
    interrupt: type[BaseException],
) -> None:
    # An operator's Ctrl-C during an import is not a plugin's finding.
    def stop() -> object:
        raise interrupt

    fake = SimpleNamespace(name="slow", value="slow:Plugin", dist=None, load=stop)
    entries = cast("list[EntryPoint]", [fake])
    with pytest.raises(interrupt):
        discover(entries)
    with pytest.raises(interrupt):
        load("slow", entries)
