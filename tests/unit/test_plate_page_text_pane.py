"""The page's text pane: what a browser is sent for a text, and that every pane of the page is one (FM-271, #657).

Panel builds a ``pn.pane.Str`` as ``'<pre>' + str(obj) + '</pre>'`` and the browser parses it,
so a ``<b>`` in a text is an element, ``from_run_doc=<its run document>`` loses its words and an
``<img src=x onerror=...>`` is live (read in Chromium, #625; the one-off look of #657 on the
served page: ``The last calibration failed: ... (nothing moves) BOLD``, three elements in the
``<pre>`` and the ``onerror`` run). ``TextPane`` escapes the text where Panel builds the
``<pre>``, and every test here reads what the browser is sent through ``browser_text``
(``tests/conftest.py``): the Bokeh model of the pane, unescaped once and parsed, never the
pane's type and never ``.object``, which keeps the raw text.

``_transform_object`` is a private method of Panel's ``Str``. The render path was measured on
Panel 1.9.5 (what a py3.10 resolve of ``panel>=1.9,<2`` installs), 1.10.0 and 1.11.1, at the
Bokeh model; a browser was looked at on 1.11.1 only (Chromium 159). A Panel that renames the
method fails ``test_a_text_pane_needs_the_method_of_panels_str_it_overrides`` first, with its
version in the message.

Panel's warnings are errors here (FM-147).
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("SMC_REQUIRE_UI"):
    import panel as pn  # CI must fail, not skip, when the extra is missing
else:
    pn = pytest.importorskip("panel")

from bokeh.document import Document

import smc.ui
from smc.ui.plate_page import TextPane, _wrapped_text

pytestmark = pytest.mark.filterwarnings(
    "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

UI_FOLDER = Path(smc.ui.__file__).parent

#: Texts the page can be handed: a stand's label, a tool's words with a placeholder, a path.
TEXTS = [
    pytest.param("<b>x</b><img src=x onerror=alert(1)>", id="element-and-onerror"),
    pytest.param(
        "reuse that calibration with from_run_doc=<its run document> (nothing moves)",
        id="placeholder-in-angle-brackets",
    ),
    pytest.param("a & b \"q\" 'q' > 3 < 4", id="ampersand-quotes-and-comparison"),
    # A text that already looks escaped is shown as itself, not unescaped on the way.
    pytest.param("&amp; &lt;b&gt; &#60;", id="already-looks-escaped"),
    pytest.param("</pre><script>alert(1)</script>", id="closes-the-pre"),
    pytest.param("<!-- c -->", id="comment"),
    pytest.param("line one\nline two  indented", id="lines-and-spaces"),
    pytest.param(
        "Objective: 40\N{MULTIPLICATION SIGN} \N{MICRO SIGN}m", id="non-ascii"
    ),
]


@pytest.mark.parametrize("text", TEXTS)
def test_a_text_pane_sends_the_browser_one_pre_element_whatever_the_text_holds(
    text: str, browser_text: Callable[[Any], str]
) -> None:
    """The text is the reader's text, at construction and after an update; ``.object`` stays raw."""
    pane = TextPane(text)
    assert browser_text(pane) == text
    assert pane.object == text

    pane.object = "first"
    assert browser_text(pane) == "first"
    pane.object = text
    assert browser_text(pane) == text
    assert pane.object == text


@pytest.mark.parametrize("blank", ["", None])
def test_a_text_pane_for_no_text_sends_panels_blank_pre(
    blank: str | None, browser_text: Callable[[Any], str]
) -> None:
    """Panel's ``<pre> </pre>`` keeps the blank line's height; nothing is escaped into it."""
    assert browser_text(TextPane(blank)) == " "
    pane = TextPane("some text")
    pane.object = blank
    assert browser_text(pane) == " "


@pytest.mark.parametrize("text", ["plain text 40x", "two\nlines", "", None])
def test_a_text_pane_sends_what_panels_str_sends_for_a_text_with_nothing_to_escape(
    text: str | None,
) -> None:
    """Panel's ``<pre>`` wrapper and its own transport escape are kept, not rebuilt."""

    def sent(pane: Any) -> str:
        return str(pane.clone().get_root(Document()).text)

    assert sent(TextPane(text)) == sent(pn.pane.Str(text))


def test_a_text_pane_is_never_chosen_by_panel_for_an_object_nobody_claims() -> None:
    """A subclass with ``Str``'s priority would take ``pn.panel(obj)`` from ``Str`` (equal priorities go to the newest).

    ``priority = -1`` keeps ``Str`` in front, and a text given to ``pn.panel`` is still Markdown.
    """
    assert type(pn.panel(object())) is pn.pane.Str
    assert type(pn.panel("text")) is pn.pane.Markdown


def test_a_text_pane_needs_the_method_of_panels_str_it_overrides() -> None:
    """The escape is in a private method of Panel: a Panel that renames it must fail here, naming itself."""
    assert hasattr(pn.pane.Str, "_transform_object"), (
        f"Panel {pn.__version__} has no pn.pane.Str._transform_object: TextPane would "
        "escape nothing (FM-271)"
    )
    assert "_transform_object" in vars(TextPane)
    for measured in ("1.9.5", "1.10.0", "1.11.1"):
        assert measured in (__doc__ or "")


def test_browser_text_refuses_a_pane_that_holds_markup(
    browser_text: Callable[[Any], str],
) -> None:
    """The helper can fail: a pane that is not text gives its elements, whatever Panel does with a ``Str``."""
    with pytest.raises(AssertionError, match="div"):
        browser_text(pn.pane.HTML("<div>x</div>"))
    with pytest.raises(AssertionError, match="img"):
        browser_text(pn.pane.Str("<img src=x onerror=1>"))


def test_a_wrapped_text_is_a_text_pane() -> None:
    """The one helper that makes 91 of the page's panes makes the escaping kind."""
    assert type(_wrapped_text("a <b> text")) is TextPane


def _pane_constructors(tree: ast.AST) -> list[str]:
    """The ``<Name>`` of every ``pn.pane.<Name>(...)`` call in ``tree``."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "pane"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pn"
        ):
            found.append(func.attr)
    return found


def test_the_page_builds_its_panes_from_text_pane_bokeh_and_the_one_html_pane() -> None:
    """A ``Str`` built anywhere in the UI package would leak: only the figures and the file links are not text.

    ``plate_page.py`` is the one module that imports Panel (as ``pn``); the pane kinds it
    builds are exactly the Bokeh figures and ``files_text``, the one HTML pane (m5 §17.12 D21,
    whose every text ``files_html`` escapes), and the ``TextPane`` class itself is the only
    other way to a ``Str``. No module of the package takes a Panel name by ``from panel ...``
    (``from panel.pane import Str`` would be a ``Str`` this test cannot see), and none builds
    a ``StaticText`` widget (Panel renders that one's value as HTML too).
    """
    modules = sorted(UI_FOLDER.glob("*.py"))
    assert UI_FOLDER / "plate_page.py" in modules
    for module in modules:
        source = module.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module))
        constructors = set(_pane_constructors(tree))
        if module.name == "plate_page.py":
            assert constructors == {"Bokeh", "HTML"}, (
                f"{module.name} builds pn.pane.{sorted(constructors)}: a text goes in "
                "TextPane, which escapes it (FM-271)"
            )
        else:
            assert not constructors, f"{module.name} builds pn.pane.{constructors}"
        imported = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "panel"
        ]
        assert not imported, f"{module.name} imports from {imported}"
        assert "StaticText" not in source, f"{module.name} builds a StaticText"
