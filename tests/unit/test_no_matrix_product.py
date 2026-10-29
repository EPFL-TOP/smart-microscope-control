"""No module under ``src/smc`` takes a matrix product (FM-131): the check is on the form, not on a warning.

``(N, 2) @ (2, 2)`` on finite coordinates raised ``RuntimeWarning: divide by zero
encountered in matmul`` from N = 1999 on numpy 2.2.6 (Python 3.10, macOS
arm64). numpy 2.5.3 on Python 3.13, the maintainer's, does not warn at any N
measured up to a million, so no test input can fail here: a ``filterwarnings``
would be green on the one machine that matters least. The syntax tree is looked
at instead, so a reintroduced ``@``, ``.dot`` or ``.matmul`` fails everywhere.

The tree and not the text, so that ``M @ (d_col, d_row)`` in a docstring (the
matrix's convention is written that way in ``geometry.py``), a comment or a
decorator is not a hit. Not seen: ``np.einsum``, ``np.tensordot``, ``np.inner``,
``np.vdot`` and ``np.linalg.multi_dot``, which no module takes today, and a
function looked up by name at run time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import smc

SRC = Path(smc.__file__).resolve().parent

#: Module (POSIX path relative to ``src/smc``) to why it may take a matrix product.
#: A legal product on something that is not a coordinate array goes here with its
#: reason, not into a weaker check. Empty: the entries are written out instead.
ALLOWED: dict[str, str] = {}

_PRODUCT_NAMES = frozenset({"dot", "matmul"})


def _hits(source: str) -> list[tuple[int, str]]:
    """``(line, form)`` of every matrix product in ``source``, in line order."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.append((node.lineno, "@"))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult):
            found.append((node.lineno, "@="))
        elif isinstance(node, ast.Attribute) and node.attr in _PRODUCT_NAMES:
            found.append((node.lineno, f".{node.attr}"))
        elif isinstance(node, ast.ImportFrom):
            found.extend(
                (node.lineno, f"import {alias.name}")
                for alias in node.names
                if alias.name in _PRODUCT_NAMES
            )
    return sorted(found)


def test_no_module_under_src_takes_a_matrix_product() -> None:
    modules = sorted(SRC.rglob("*.py"))
    assert modules  # a walk that finds nothing would pass for the wrong reason

    found: list[str] = []
    for module in modules:
        relative = module.relative_to(SRC).as_posix()
        if relative in ALLOWED:
            continue
        source = module.read_text(encoding="utf-8")
        found.extend(
            f"src/smc/{relative}:{line} {form}" for line, form in _hits(source)
        )

    assert not found, (
        "a matrix product on coordinates warns on numpy 2.2.6 (FM-131): write the "
        "entries out, or give the module an ALLOWED entry with its reason\n"
        + "\n".join(found)
    )


@pytest.mark.parametrize(
    ("source", "form"),
    [
        ("c = a @ b", "@"),
        ("a @= b", "@="),
        ("c = np.dot(a, b)", ".dot"),
        ("c = np.matmul(a, b)", ".matmul"),
        ("c = a.dot(b)", ".dot"),
        ("from numpy import dot", "import dot"),
    ],
)
def test_the_check_finds_each_form_of_a_matrix_product(source: str, form: str) -> None:
    assert [found for _, found in _hits(source)] == [form]


def test_the_check_ignores_a_decorator() -> None:
    source = (
        "import functools\n"
        "\n"
        "\n"
        "class Thing:\n"
        "    @functools.cache\n"
        "    def size(self) -> int:\n"
        "        return 1\n"
        "\n"
        "    @property\n"
        "    def name(self) -> str:\n"
        "        return 'thing'\n"
    )

    assert _hits(source) == []


def test_the_check_ignores_a_docstring_and_a_comment() -> None:
    source = (
        '"""``(dx, dy) = M @ (d_col, d_row)``, and ``a.dot(b)``."""\n'
        "# the product a @ b is written out below\n"
        "x = 1\n"
    )

    assert _hits(source) == []
