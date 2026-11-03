import ast
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

import smc

SMC = Path(smc.__file__).resolve().parent
PLUGINS = SMC / "plugins"

#: The two top-level packages of ``smc`` that may name ``smc.testing`` at import time:
#: the stand-ins themselves, and the plugins, whose rule is the stricter one below.
STAND_INS = ("testing", "plugins")


def _package_of(path: Path) -> list[str]:
    """The dotted package a relative import in ``path`` is resolved against: the directory it is in.

    ``with_suffix("")`` leaves ``__init__`` as the last part of a package's own file,
    and it is dropped like the module's name: ``smc/ui/__init__.py`` and
    ``smc/ui/controller.py`` are both in ``smc.ui``.
    """
    return list(path.relative_to(SMC.parent).with_suffix("").parts)[:-1]


def test_version_is_exposed() -> None:
    assert smc.__version__
    assert smc.__version__ != "0.0.0+unknown"


def _import_statements(
    tree: ast.AST, *, inside_functions: bool
) -> Iterator[ast.Import | ast.ImportFrom]:
    """The ``import`` statements of ``tree``; those under a ``def`` only when ``inside_functions``.

    A statement in a class body, a ``try:`` or an ``if:`` (``TYPE_CHECKING``
    included) runs when the module is imported, so it is always found.
    """
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield node
        if not inside_functions and isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
        ):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _lines_importing_smc_testing(
    source: str, package: Sequence[str], *, inside_functions: bool = True
) -> list[int]:
    """The lines of ``source`` that import ``smc.testing``, or a name from it.

    Every ``import`` statement counts, the ones under ``TYPE_CHECKING`` and
    inside functions included: a plugin that names the test package even for a
    type has made the package a dependency of a stand. A relative import is
    resolved against ``package``, the dotted package of the file.
    ``inside_functions=False`` leaves out the statements under a ``def``, which
    run when the function is called, not when the module is imported.
    """
    lines: list[int] = []
    for node in _import_statements(
        ast.parse(source), inside_functions=inside_functions
    ):
        if isinstance(node, ast.Import):
            targets = [alias.name.split(".") for alias in node.names]
        else:
            base = (
                list(package[: len(package) - (node.level - 1)]) if node.level else []
            )
            module = [*base, *(node.module.split(".") if node.module else [])]
            # `from smc import testing` names the module in an alias.
            targets = [module, *([*module, alias.name] for alias in node.names)]
        if any(target[:2] == ["smc", "testing"] for target in targets):
            lines.append(node.lineno)
    return sorted(lines)


def _module_level_lines_importing_smc_testing(
    source: str, package: Sequence[str]
) -> list[int]:
    """The lines of ``source`` that load ``smc.testing`` when the module is imported."""
    return _lines_importing_smc_testing(source, package, inside_functions=False)


@pytest.mark.parametrize(
    "source",
    [
        "import smc.testing",
        "import numpy, smc.testing.synthetic as synthetic",
        "from smc.testing import FakeCore",
        "from smc.testing.synthetic import Blob",
        "from smc import testing",
        "from smc import hardware, testing",
        "if TYPE_CHECKING:\n    from smc.testing import FakeCore",
        "def f():\n    from smc.testing.sample_stand import install_sample",
        "from ...testing import FakeCore",
        "from ... import testing",
    ],
)
def test_the_import_check_sees_every_way_to_import_smc_testing(source: str) -> None:
    # The check below passes when nothing is found: each form it must find is
    # shown here, for a file of the package smc.plugins.cells (FM-43).
    assert _lines_importing_smc_testing(source, ["smc", "plugins", "cells"]) != []


@pytest.mark.parametrize(
    "source",
    [
        "import smc",
        "from smc.hardware import Microscope",
        "from smc import hardware",
        "import numpy.testing",
        "from numpy.testing import assert_allclose",
        "from smc import testing_helpers",
        "from . import testing",
        "from ...hardware import Microscope",
    ],
)
def test_the_import_check_leaves_other_imports_alone(source: str) -> None:
    assert _lines_importing_smc_testing(source, ["smc", "plugins", "cells"]) == []


def test_no_plugin_imports_smc_testing() -> None:
    # smc.testing is the fakes and the demo stand: a plugin that imports it would
    # load it on every real stand (m1 §8, "The demo-sample stand", #260), and a
    # plugin knows no stand. The substitution lives in the facade, imported inside
    # the call that needs it.
    files = sorted(PLUGINS.rglob("*.py"))
    assert len(files) > 10  # a glob that finds nothing would pass by not looking
    offenders = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        offenders += [
            f"{path.relative_to(PLUGINS)}:{line}"
            for line in _lines_importing_smc_testing(source, _package_of(path))
        ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "import smc.testing",
        "from smc.testing.sample_stand import sample_refs",
        "from smc import testing",
        "if TYPE_CHECKING:\n    from smc.testing import FakeCore",
        "try:\n    from smc.testing import FakeCore\nexcept ImportError:\n    pass",
        "class Stand:\n    from smc.testing import FakeCore",
    ],
)
def test_the_module_level_import_check_sees_every_way_to_hoist_the_import(
    source: str,
) -> None:
    # The rule below passes when nothing is found: each way to load `smc.testing`
    # when a module is imported is shown here (FM-43).
    assert _module_level_lines_importing_smc_testing(source, ["smc", "ui"]) != []


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    from smc.testing.sample_stand import sample_refs",
        "async def f():\n    from smc.testing.sample_stand import sample_refs",
        "class Stand:\n    def f(self):\n        import smc.testing",
        "import smc\nfrom smc.hardware import Microscope",
    ],
)
def test_the_module_level_import_check_leaves_an_import_inside_a_function_alone(
    source: str,
) -> None:
    # The sanctioned way to reach the stand-ins: inside the call that needs them.
    assert _module_level_lines_importing_smc_testing(source, ["smc", "ui"]) == []


@pytest.mark.parametrize(
    ("relative", "package"),
    [
        ("ui/controller.py", ["smc", "ui"]),
        ("ui/__init__.py", ["smc", "ui"]),
        ("__init__.py", ["smc"]),
        ("plugins/cells/plugin.py", ["smc", "plugins", "cells"]),
        ("plugins/cells/__init__.py", ["smc", "plugins", "cells"]),
    ],
)
def test_the_package_of_a_file_is_the_directory_it_is_in(
    relative: str, package: list[str]
) -> None:
    assert _package_of(SMC / relative) == package


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("__init__.py", "from .testing import FakeCore"),
        ("__init__.py", "from . import testing"),
        ("ui/__init__.py", "from ..testing import FakeCore"),
        ("ui/__init__.py", "from .. import testing"),
        ("ui/controller.py", "from ..testing import FakeCore"),
        ("hardware/backends/__init__.py", "from ...testing import FakeCore"),
    ],
)
def test_the_module_level_import_check_sees_a_relative_hoist_in_a_packages_init(
    relative: str, source: str
) -> None:
    # Relative re-exports are what an `__init__.py` is made of: the package that the
    # rule resolves them against is that of the file's own directory (FM-43, FM-176).
    package = _package_of(SMC / relative)
    assert _module_level_lines_importing_smc_testing(source, package) != []


def test_no_module_loads_smc_testing_at_import() -> None:
    # `smc.testing` stands in for a stand (FakeCore, a synthetic plate): a module that
    # loads it when it is imported loads it on every real stand, however it is used
    # (m1 §8, "The demo-sample stand"; FM-176). A call that needs it imports it inside
    # itself. The plugins have the stricter rule above, and `smc/testing` is the
    # stand-in itself.
    files = sorted(
        path
        for path in SMC.rglob("*.py")
        if path.relative_to(SMC).parts[0] not in STAND_INS
    )
    assert len(files) > 20  # a glob that finds nothing would pass by not looking
    offenders = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        offenders += [
            f"{path.relative_to(SMC)}:{line}"
            for line in _module_level_lines_importing_smc_testing(
                source, _package_of(path)
            )
        ]
    assert offenders == []
