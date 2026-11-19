"""The registry of selectors and orderers (algorithms design §5.4, §5.8, #567): findings, never a crash.

The test algorithms are classes of this module, loaded through
``EntryPoint(value=f"{__name__}:<Class>")``: a module already in ``sys.modules``
needs no installed distribution. ``registry.entry_points`` is replaced by a
function that takes the group by keyword, as ``tests/test_cli_presets.py`` does
for the plugins, so that the tests never depend on what is installed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from typing import Any

import pytest

from smc.selection import registry
from smc.selection.api import (
    Choice,
    OrdererParams,
    SelectionError,
    SelectionInput,
    SelectorParams,
)
from smc.selection.registry import (
    ORDERER_GROUP,
    SELECTOR_GROUP,
    orderer,
    orderer_names,
    orderers,
    selector,
    selector_names,
    selectors,
)

THIS = __name__


class GoodParams(SelectorParams):
    pass


class GoodOrdererParams(OrdererParams):
    pass


class Good:
    name = "good"
    summary = "Chooses nothing, correctly."
    Params = GoodParams

    def select(self, data: SelectionInput, params: GoodParams) -> Sequence[Choice]:
        return []


class Another(Good):
    name = "another"


class GoodOrderer:
    name = "good-orderer"
    summary = "Keeps the order, correctly."
    Params = GoodOrdererParams

    def order(
        self, choices: Sequence[Choice], data: SelectionInput, params: Any
    ) -> Sequence[int]:
        return list(range(len(choices)))


class AnotherOrderer(GoodOrderer):
    name = "another-orderer"


class NamedOtherwise(Good):
    name = "not-the-entry"


class NotKebab(Good):
    name = "Not_Kebab"


class NoSummary(Good):
    name = "no-summary"
    summary = None  # type: ignore[assignment]


class EmptySummary(Good):
    name = "empty-summary"
    summary = "   "


class TwoLinesSummary(Good):
    name = "two-lines"
    summary = "First line.\nSecond line."


class RaisingMember(Good):
    name = "raising-member"

    @property
    def summary(self) -> str:  # type: ignore[override]
        raise RuntimeError("needs a file that is not there")


class ParamsOfThePlugins(Good):
    name = "plugin-params"
    Params = dict  # type: ignore[assignment]


class ParamsOfTheOtherKind(Good):
    name = "orderer-params"
    Params = GoodOrdererParams  # type: ignore[assignment]


class NoSelect:
    name = "no-select"
    summary = "Has nothing to run."
    Params = GoodParams


class SelectIsNotCallable(Good):
    name = "not-callable"
    select = None  # type: ignore[assignment]


class NeedsAnArgument(Good):
    name = "needs-argument"

    def __init__(self, path: str) -> None:
        self.path = path


class FailsToCreate(Good):
    name = "fails-to-create"

    def __init__(self) -> None:
        raise ValueError("line one\nline two")


class Interrupts(Good):
    name = "interrupts"

    def __init__(self) -> None:
        raise KeyboardInterrupt


class OrdererWithoutOrder:
    name = "no-order"
    summary = "Has nothing to run."
    Params = GoodOrdererParams


class OrdererWithSelectorParams(GoodOrderer):
    name = "selector-params"
    Params = GoodParams  # type: ignore[assignment]


class NotAClass:
    pass


NOT_A_CLASS = NotAClass()


def _entry(name: str, target: str, group: str = SELECTOR_GROUP) -> EntryPoint:
    return EntryPoint(name=name, value=f"{THIS}:{target}", group=group)


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[EntryPoint]]:
    """The two groups as a test sets them: a dict that ``registry.entry_points`` reads at each call."""
    groups: dict[str, list[EntryPoint]] = {SELECTOR_GROUP: [], ORDERER_GROUP: []}
    monkeypatch.setattr(registry, "entry_points", lambda *, group: list(groups[group]))
    return groups


def test_a_registered_selector_and_orderer_are_found_by_name_and_listed_sorted(
    installed: dict[str, list[EntryPoint]],
) -> None:
    installed[SELECTOR_GROUP] += [_entry("good", "Good"), _entry("another", "Another")]
    installed[ORDERER_GROUP] += [
        _entry("good-orderer", "GoodOrderer", ORDERER_GROUP),
        _entry("another-orderer", "AnotherOrderer", ORDERER_GROUP),
    ]

    assert selector_names() == ("another", "good")
    assert orderer_names() == ("another-orderer", "good-orderer")
    assert list(selectors()) == ["another", "good"]
    assert list(orderers()) == ["another-orderer", "good-orderer"]
    assert isinstance(selector("good"), Good)
    assert isinstance(orderer("good-orderer"), GoodOrderer)
    # Each is a fresh instance with no argument, the group's own: a selector is not an orderer.
    assert selector("good") is not selector("good")
    with pytest.raises(SelectionError, match="no orderer named 'good'"):
        orderer("good")
    with pytest.raises(SelectionError, match="no selector named 'good-orderer'"):
        selector("good-orderer")


def test_an_unknown_name_lists_the_known_ones_sorted_without_importing_anything(
    installed: dict[str, list[EntryPoint]],
) -> None:
    installed[SELECTOR_GROUP] += [
        _entry("zeta", "Good"),
        EntryPoint(
            name="broken",
            value="smc_selection_no_such_module:Thing",
            group=SELECTOR_GROUP,
        ),
        _entry("good", "Good"),
    ]

    # The names are read, not imported: the module that does not exist is listed.
    assert selector_names() == ("broken", "good", "zeta")
    with pytest.raises(
        SelectionError, match="no selector named 'nope'; known: broken, good, zeta"
    ):
        selector("nope")
    with pytest.raises(SelectionError, match="no orderer is registered"):
        orderer("nope")


def test_a_lookup_imports_only_the_entries_of_that_name(
    installed: dict[str, list[EntryPoint]], caplog: pytest.LogCaptureFixture
) -> None:
    installed[SELECTOR_GROUP] += [
        EntryPoint(
            name="broken",
            value="smc_selection_no_such_module:Thing",
            group=SELECTOR_GROUP,
        ),
        _entry("good", "Good"),
    ]

    with caplog.at_level(logging.WARNING, logger="smc.selection"):
        found = selector("good")

    assert found.name == "good"
    assert caplog.records == []


@pytest.mark.parametrize(
    ("entry_name", "target", "reason"),
    [
        (
            "good",
            "NamedOtherwise",
            "is named 'not-the-entry', but its entry point is named 'good'",
        ),
        ("Not_Kebab", "NotKebab", "'Not_Kebab', which is not kebab-case"),
        ("no-summary", "NoSummary", "has a summary that is not a string: None"),
        ("empty-summary", "EmptySummary", "has an empty summary"),
        ("two-lines", "TwoLinesSummary", "has a summary of more than one line"),
        (
            "raising-member",
            "RaisingMember",
            "could not be inspected: RuntimeError: needs a file that is not there",
        ),
        (
            "plugin-params",
            "ParamsOfThePlugins",
            "has Params dict, which is not a subclass of SelectorParams",
        ),
        (
            "orderer-params",
            "ParamsOfTheOtherKind",
            "has Params GoodOrdererParams, which is not a subclass of SelectorParams",
        ),
        ("no-select", "NoSelect", "lacks select"),
        ("not-callable", "SelectIsNotCallable", "has a select that is not callable"),
        (
            "needs-argument",
            "NeedsAnArgument",
            "could not be created with no argument: TypeError",
        ),
        (
            "fails-to-create",
            "FailsToCreate",
            "could not be created with no argument: ValueError: line one line two",
        ),
        ("good", "Missing", "could not be imported: AttributeError"),
        ("good", "NOT_A_CLASS", "names a NotAClass, not a class"),
    ],
)
def test_a_selector_that_breaks_a_rule_is_a_finding_not_an_exception_of_the_listing(
    installed: dict[str, list[EntryPoint]],
    caplog: pytest.LogCaptureFixture,
    entry_name: str,
    target: str,
    reason: str,
) -> None:
    installed[SELECTOR_GROUP] += [
        _entry("another", "Another"),
        _entry(entry_name, target),
        EntryPoint(
            name="gone",
            value="smc_selection_no_such_module:Thing",
            group=SELECTOR_GROUP,
        ),
    ]

    with caplog.at_level(logging.WARNING, logger="smc.selection"):
        listed = selectors()

    # The others load, and each that does not is one warning with its reason.
    assert "another" in listed
    assert entry_name not in listed
    assert "gone" not in listed
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any(entry_name in w and reason in w for w in warnings), warnings
    assert any("gone" in w and "ModuleNotFoundError" in w for w in warnings)
    # By name, the reason is the error: the person who asked for it reads why.
    with pytest.raises(SelectionError) as found:
        selector(entry_name)
    assert f"selector {entry_name!r} " in str(found.value)
    assert reason in str(found.value)
    assert "\n" not in str(found.value)
    # The reason is text and the error carries no traceback of the algorithm (FM-98).
    assert found.value.__cause__ is None
    assert found.value.__context__ is None


def test_an_orderer_that_breaks_a_rule_is_a_finding_like_a_selector(
    installed: dict[str, list[EntryPoint]], caplog: pytest.LogCaptureFixture
) -> None:
    installed[ORDERER_GROUP] += [
        _entry("good-orderer", "GoodOrderer", ORDERER_GROUP),
        _entry("no-order", "OrdererWithoutOrder", ORDERER_GROUP),
        _entry("selector-params", "OrdererWithSelectorParams", ORDERER_GROUP),
    ]

    with caplog.at_level(logging.WARNING, logger="smc.selection"):
        listed = orderers()

    assert list(listed) == ["good-orderer"]
    assert len(caplog.records) == 2
    with pytest.raises(SelectionError, match="orderer 'no-order' lacks order"):
        orderer("no-order")
    with pytest.raises(
        SelectionError,
        match="orderer 'selector-params' has Params GoodParams, which is not a "
        "subclass of OrdererParams",
    ):
        orderer("selector-params")


def test_two_entries_of_one_name_with_different_targets_are_both_refused(
    installed: dict[str, list[EntryPoint]], caplog: pytest.LogCaptureFixture
) -> None:
    installed[SELECTOR_GROUP] += [
        _entry("good", "Good"),
        EntryPoint(name="good", value=f"{THIS}:NamedOtherwise", group=SELECTOR_GROUP),
        _entry("another", "Another"),
    ]

    with caplog.at_level(logging.WARNING, logger="smc.selection"):
        listed = selectors()

    # Neither shadows the other, and the name is listed once among the names.
    assert list(listed) == ["another"]
    assert selector_names() == ("another", "good")
    with pytest.raises(
        SelectionError,
        match=rf"selector 'good' is registered twice, with different targets: "
        rf"{THIS}:Good .*{THIS}:NamedOtherwise",
    ):
        selector("good")


def test_an_editable_install_listing_one_entry_twice_is_one_entry(
    installed: dict[str, list[EntryPoint]], caplog: pytest.LogCaptureFixture
) -> None:
    installed[SELECTOR_GROUP] += [_entry("good", "Good"), _entry("good", "Good")]

    with caplog.at_level(logging.WARNING, logger="smc.selection"):
        listed = selectors()

    assert list(listed) == ["good"]
    assert caplog.records == []
    assert selector_names() == ("good",)
    assert selector("good").name == "good"


def test_no_selector_at_all_says_to_reinstall(
    installed: dict[str, list[EntryPoint]],
) -> None:
    with pytest.raises(
        SelectionError, match=r"no selector is registered.*pip install -e \."
    ):
        selector("most-cells")
    with pytest.raises(
        SelectionError, match=r"no orderer is registered.*pip install -e \."
    ):
        orderer("nearest")
    assert selector_names() == ()
    assert selectors() == {}
    # With some registered, the text lists them and does not talk of installing.
    installed[SELECTOR_GROUP] += [_entry("good", "Good")]
    with pytest.raises(SelectionError) as found:
        selector("most-cells")
    assert "pip install" not in str(found.value)
    assert "known: good" in str(found.value)


def test_the_built_ins_are_registered_in_the_installed_metadata() -> None:
    # The real `entry_points`, no replacement: it fails until smc is installed
    # again after the entry points were added (FM-112), as the plugin registry's
    # own test of the installed plugins does. Membership only, never the whole
    # set: a lab's own package adds to it (FM-191).
    assert "most-cells" in selector_names()
    assert {"nearest", "as-chosen"} <= set(orderer_names())
    assert selector("most-cells").name == "most-cells"
    assert orderer("nearest").name == "nearest"
    assert orderer("as-chosen").name == "as-chosen"
    assert {"most-cells"} <= set(selectors())
    assert {"nearest", "as-chosen"} <= set(orderers())


def test_a_keyboard_interrupt_while_an_algorithm_is_created_is_not_a_finding(
    installed: dict[str, list[EntryPoint]],
) -> None:
    installed[SELECTOR_GROUP] += [_entry("interrupts", "Interrupts")]

    with pytest.raises(KeyboardInterrupt):
        selector("interrupts")
    with pytest.raises(KeyboardInterrupt):
        selectors()


def test_the_registry_reads_the_group_by_keyword_so_that_a_test_can_replace_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def entry_points(*, group: str) -> list[EntryPoint]:
        asked.append(group)
        return []

    monkeypatch.setattr(registry, "entry_points", entry_points)

    selector_names()
    orderer_names()

    assert asked == ["smc.selectors", "smc.orderers"]
