import ast
from collections.abc import Sequence
from pathlib import Path

import pytest

import smc

PLUGINS = Path(smc.__file__).resolve().parent / "plugins"


def test_version_is_exposed() -> None:
    assert smc.__version__
    assert smc.__version__ != "0.0.0+unknown"


def _lines_importing_smc_testing(source: str, package: Sequence[str]) -> list[int]:
    """The lines of ``source`` that import ``smc.testing``, or a name from it.

    Every ``import`` statement counts, the ones under ``TYPE_CHECKING`` and
    inside functions included: a plugin that names the test package even for a
    type has made the package a dependency of a stand. A relative import is
    resolved against ``package``, the dotted package of the file.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = (
                list(package[: len(package) - (node.level - 1)]) if node.level else []
            )
            module = [*base, *(node.module.split(".") if node.module else [])]
            # `from smc import testing` names the module in an alias.
            targets = [module, *([*module, alias.name] for alias in node.names)]
        else:
            continue
        if any(target[:2] == ["smc", "testing"] for target in targets):
            lines.append(node.lineno)
    return lines


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
        parts = list(path.relative_to(PLUGINS.parent.parent).with_suffix("").parts)
        package = parts if path.name == "__init__.py" else parts[:-1]
        source = path.read_text(encoding="utf-8")
        offenders += [
            f"{path.relative_to(PLUGINS)}:{line}"
            for line in _lines_importing_smc_testing(source, package)
        ]
    assert offenders == []
