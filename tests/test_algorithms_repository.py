"""The files that set a collaborator up work (algorithms design §7, #554): the label, the template, CODEOWNERS and the skill's templates.

None of them is code that a test of the kit reaches, and each fails quietly: GitHub
drops a label an issue form names but the repository lacks, a CODEOWNERS line for a
path that was renamed protects nothing, and a skill whose template stopped being
valid gives the collaborator a first file that does not load.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from smc.selection import registry

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "new-selector" / "SKILL.md"


def _label_names() -> set[str]:
    text = (ROOT / "scripts" / "github" / "labels.json").read_text(encoding="utf-8")
    return {label["name"] for label in json.loads(text)}


def test_the_algorithm_template_names_labels_the_repository_has() -> None:
    template = ROOT / ".github" / "ISSUE_TEMPLATE" / "algorithm.yml"
    match = re.search(
        r"^labels: \[(.*)\]\s*$", template.read_text(encoding="utf-8"), re.M
    )
    assert match, "the template has no `labels: [...]` line"
    labels = {item.strip().strip("\"'") for item in match.group(1).split(",")}
    assert labels == {"type: feature", "area: algorithms"}
    assert labels <= _label_names(), labels - _label_names()


def test_every_issue_template_names_labels_the_repository_has() -> None:
    for template in sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml")):
        match = re.search(
            r"^labels: \[(.*)\]\s*$", template.read_text(encoding="utf-8"), re.M
        )
        if match is None:
            continue
        labels = {item.strip().strip("\"'") for item in match.group(1).split(",")}
        assert labels <= _label_names(), f"{template.name}: {labels - _label_names()}"


def test_the_algorithm_template_asks_what_the_design_says_it_asks() -> None:
    text = (ROOT / ".github" / "ISSUE_TEMPLATE" / "algorithm.yml").read_text(
        encoding="utf-8"
    )
    for label in ("Goal", "Data in", "Decision out", "How it is judged"):
        block = re.search(
            rf"label: {label}\n(?:.*\n)*?\s+validations:\n\s+required: true", text
        )
        assert block, f"{label} is not a required field"
    assert "label: The wiring issue" in text
    assert "dropdown" not in text, "an algorithm never needs a stand"


def test_every_codeowners_path_exists() -> None:
    lines = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
    rules = [
        line.split() for line in lines if line.strip() and not line.startswith("#")
    ]
    assert {rule[0] for rule in rules} >= {"/src/smc/hardware/", "/profiles/"}
    for pattern, *owners in rules:
        assert owners, f"{pattern} has no owner"
        assert (ROOT / pattern.lstrip("/")).exists(), (
            f"{pattern} is not in the repository"
        )


def _skill_blocks(language: str) -> list[str]:
    text = SKILL.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```", text, flags=re.S | re.M)


def _filled(template: str) -> str:
    """The skill's placeholders replaced for a selector named ``dense-fields``."""
    return (
        template.replace("ClassNameParams", "DenseFieldsParams")
        .replace("ClassName", "DenseFields")
        .replace("kebab-name", "dense-fields")
        .replace("smc.selection.snake_name", "dense_fields")
        .replace("snake_name", "dense_fields")
    )


def test_the_skill_templates_make_a_selector_whose_own_tests_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, test = _skill_blocks("python")
    (tmp_path / "dense_fields.py").write_text(_filled(module), encoding="utf-8")
    (tmp_path / "test_selector_dense_fields.py").write_text(
        _filled(test), encoding="utf-8"
    )
    # Inherited environment, and no -I: a mutant's copy of src is first on PYTHONPATH.
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 passed" in done.stdout, done.stdout

    # What the entry-point line promises: the registry loads the class by name.
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (
            [
                EntryPoint(
                    name="dense-fields",
                    value="dense_fields:DenseFields",
                    group=group,
                )
            ]
            if group == registry.SELECTOR_GROUP
            else []
        ),
    )
    try:
        loaded = registry.selector("dense-fields")
    finally:
        sys.modules.pop("dense_fields", None)
    assert loaded.name == "dense-fields"


def test_the_skill_entry_point_line_and_page_headings_are_the_designs() -> None:
    (entry,) = _skill_blocks("toml")
    assert entry.strip() == 'kebab-name = "smc.selection.snake_name:ClassName"'
    (page,) = _skill_blocks("markdown")
    headings = re.findall(r"^## (.*)$", page, flags=re.M)
    assert headings == ["What it measures", "Parameters", "Tried on", "Numbers"]
