"""The guard of the collaborator's kit (algorithms design §8): it loads with no hardware.

The kit is what a second developer runs on a laptop that has no Micro-Manager.
Two things keep it so, and each fails without the other noticing: a subprocess
that imports the kit and looks at what got loaded (an import that goes through
a third module is seen), and a walk over the kit's own source (an import that
is only made inside a function, or that a test never reaches, is seen).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

#: What the subprocess imports. The selection package is there by every module,
#: and ``route``'s plugin, which imports it, is there too: a selector is loaded
#: inside the plugin, and the plugin must not pull the Micro-Manager bindings in
#: by it.
KIT_MODULES = [
    "smc.data",
    "smc.data.cli",
    "smc.cli",
    "smc.selection",
    "smc.selection.api",
    "smc.selection.registry",
    "smc.selection.checks",
    "smc.selection.builtin",
    "smc.selection.testing",
    "smc.plugins.route.plugin",
]

#: The packages of ``src/smc/`` the walk covers.
KIT_PACKAGES = ["data", "selection"]

#: What no module of the kit imports, itself or a submodule of it.
FORBIDDEN = ["smc.hardware", "smc.testing", "smc.ui", "pymmcore_plus", "pymmcore"]

SRC = Path(__file__).resolve().parents[1] / "src" / "smc"


def _loaded_hardware_modules(modules: list[str], cwd: Path) -> list[str]:
    """What of ``pymmcore_plus`` and ``pymmcore`` a fresh interpreter has loaded after importing ``modules``."""
    code = (
        "import json, sys\n"
        + "".join(f"import {name}\n" for name in modules)
        + "print(json.dumps(sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('pymmcore', 'pymmcore_plus'))))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert isinstance(loaded, list)
    return [str(name) for name in loaded]


def _is_forbidden(name: str) -> bool:
    return any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)


def _violations(path: Path) -> list[str]:
    """The imports of ``path`` that the kit does not allow, one line each, with their line number.

    ``from smc import hardware`` imports ``smc.hardware`` and is read as that;
    a relative import is refused whatever it names, because the kit's modules
    are loaded by file in a collaborator's checkout and by name in the package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    found.append(f"{path.name}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                dots = "." * node.level
                found.append(
                    f"{path.name}:{node.lineno}: relative import "
                    f"from {dots}{node.module or ''}"
                )
                continue
            module = node.module or ""
            if _is_forbidden(module):
                found.append(f"{path.name}:{node.lineno}: from {module} import ...")
                continue
            for alias in node.names:
                if _is_forbidden(f"{module}.{alias.name}"):
                    found.append(
                        f"{path.name}:{node.lineno}: from {module} import {alias.name}"
                    )
    return found


def test_kit_never_imports_pymmcore_plus(tmp_path: Path) -> None:
    # A collaborator's laptop has no Micro-Manager: importing the kit must not
    # reach for it, through smc.cli either (the sub-app is added to it).
    assert {"smc.data", "smc.data.cli", "smc.cli"} <= set(KIT_MODULES)
    assert {
        "smc.selection",
        "smc.selection.api",
        "smc.selection.registry",
        "smc.selection.checks",
        "smc.selection.builtin",
        "smc.selection.testing",
        "smc.plugins.route.plugin",
    } <= set(KIT_MODULES)
    assert _loaded_hardware_modules(KIT_MODULES, tmp_path) == []


def test_importing_the_cli_does_not_load_pillow(tmp_path: Path) -> None:
    # Pillow is a declared dependency, but only the drawing needs it: `smc dataset
    # draw` imports `smc.data.draw` inside its command, so that every other command
    # (and `import smc.data` for a collaborator's script) starts without it.
    code = (
        "import json, sys\n"
        "import smc.cli\n"
        "import smc.data\n"
        "import smc.data.cli\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.split('.')[0] == 'PIL')))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_kit_modules_never_import_the_hardware_layer() -> None:
    walked = 0
    found: list[str] = []
    for package in KIT_PACKAGES:
        files = sorted((SRC / package).rglob("*.py"))
        assert files, f"src/smc/{package} holds no module: the walk looks at nothing"
        for path in files:
            walked += 1
            found.extend(_violations(path))
    assert walked > 0
    assert found == []


def test_the_import_walk_sees_a_from_import_a_submodule_and_a_relative_import(
    tmp_path: Path,
) -> None:
    # The walk is the only thing that fails the test above if it is blind, so it
    # is shown three imports it must catch (and two it must let through).
    sources = {
        "from_import.py": "from smc import hardware\n",
        "submodule.py": "import smc.hardware.core\n",
        "relative.py": "from . import x\n",
        "nested.py": "def f():\n    from smc.testing.fakes import FakeCore\n",
        # One source for each name of FORBIDDEN that no source above has: with
        # the name taken out of the list, only its own source goes unseen.
        "ui.py": "from smc.ui import page\n",
        "bindings_plus.py": "def f():\n    import pymmcore_plus\n",
        "bindings.py": "from pymmcore import CMMCore\n",
        "clean.py": (
            "import numpy\n"
            "from smc.plugins.slots import Cell\n"
            "from smc import hardware_notes\n"
        ),
    }
    seen = {}
    for name, text in sources.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        seen[name] = _violations(path)
    assert {name: len(found) for name, found in seen.items()} == {
        "from_import.py": 1,
        "submodule.py": 1,
        "relative.py": 1,
        "nested.py": 1,
        "ui.py": 1,
        "bindings_plus.py": 1,
        "bindings.py": 1,
        "clean.py": 0,
    }, seen
    assert "from smc import hardware" in seen["from_import.py"][0]
    assert "import smc.hardware.core" in seen["submodule.py"][0]
    assert "relative import" in seen["relative.py"][0]
    assert "smc.testing" in seen["nested.py"][0]
    assert "smc.ui" in seen["ui.py"][0]
    assert "pymmcore_plus" in seen["bindings_plus.py"][0]
    assert "from pymmcore import" in seen["bindings.py"][0]
