---
name: new-selector
description: Scaffold a new selection algorithm for the algorithm developer's kit - the module with its Params and a working Selector, its test on cells placed by hand, the entry point and its page under docs/algorithms/. Argument: the selector's name in kebab-case.
---

# /new-selector <name>

For an `area: algorithms` issue. Read `docs/algorithms/README.md` first (what the kit
is, the dataset, what never to assume), then `docs/design/algorithms.md` §5 (the
contract). A selector turns a scan's cells into fields to image; an orderer turns those
fields into a visiting order. This skill writes a selector; an orderer is the same with
the `Orderer` protocol and the `smc.orderers` group (`src/smc/selection/builtin.py`
holds `Nearest` and `AsChosen`).

`<name>` is kebab-case (`dense-fields`); `<snake_name>` is the same in snake_case
(`dense_fields`), because a kebab name is not a module path; `<ClassName>` is the
CamelCase of it (`DenseFields`). In the templates below `kebab-name`, `snake_name`,
`ClassName` and `ClassNameParams` stand for them: replace every one.

Create four files and one line:

```
src/smc/selection/<snake_name>.py            the Params and the Selector
tests/test_selector_<snake_name>.py          two cases whose answer is known
docs/algorithms/<name>.md                    what it measures, its parameters, what was tried
pyproject.toml                               the entry-point line
```

**Test module basenames are unique under `tests/`** (pytest imports them by basename):
check that `tests/test_selector_<snake_name>.py` does not exist anywhere else first.
Never touch `src/smc/hardware/`, `profiles/` or `src/smc/ui/`: an `area: algorithms`
change is pure (`docs/design/algorithms.md` §7.1).

## The module

The example is **a working start to replace**, not a method anyone relies on: it keeps
the scan's planned fields that hold the most wanted cells, never two that overlap. It
reads the plan's fields and not the frames so that the same code runs on a dataset and
on cells placed by hand in a test (`make_input` has a plan and no frames); use
`data.scan.frames(well)` and `data.scan.image(frame)` when the algorithm reads pixels.

```python
"""kebab-name: <what it chooses, and why that is the decision to make>.

<One paragraph: the measure it decides on and the biology or the imaging reason
behind it. Say what is assumed, and label it as assumed.>
"""

from __future__ import annotations

from pydantic import Field

from smc.plugins.route.algorithm import first_overlap
from smc.selection import Choice, SelectionInput, SelectorParams

__all__ = ["ClassName", "ClassNameParams"]


class ClassNameParams(SelectorParams):
    """What a field must hold to be chosen."""

    min_cells: int = Field(
        1,
        ge=1,
        description="A field must hold at least this many cells of the wanted classes.",
    )


class ClassName:
    """The planned fields with the most wanted cells, best first, never overlapping."""

    name = "kebab-name"
    summary = (
        "The planned fields that hold the most cells of the wanted classes, "
        "best first, never overlapping."
    )
    Params = ClassNameParams

    def select(self, data: SelectionInput, params: ClassNameParams) -> list[Choice]:
        """Rank each well's allowed fields by the cells they hold and keep the best that do not overlap.

        Raises:
            SelectionError: The scan has no cells (``data.cells`` says why).
        """
        if not data.cells:  # raises SelectionError, with the reason, when it has none
            return []
        chosen: list[Choice] = []
        for well in data.scan.wells:
            candidates: list[tuple[int, float, float]] = []
            for planned in data.scan.plan.fields:
                if planned.well != well or not data.allowed(
                    planned.x_um, planned.y_um, well
                ):
                    continue
                held = len(data.cells_in(planned.x_um, planned.y_um, well))
                if held >= params.min_cells:
                    candidates.append((held, planned.x_um, planned.y_um))
            candidates.sort(key=lambda c: -c[0])  # a tie keeps the plan's order
            centres: list[tuple[float, float]] = []
            for held, x_um, y_um in candidates:
                if len(centres) == data.fields_per_well:
                    break
                overlap = first_overlap(
                    [*centres, (x_um, y_um)],
                    matrix=data.matrix_um_per_px,
                    image_shape_px=data.image_shape_px,
                )
                if overlap is not None:
                    continue
                centres.append((x_um, y_um))
                score = float(held)
                measures = {"cells_in_field": score}
                chosen.append(Choice(well, x_um, y_um, score=score, measures=measures))
        return chosen
```

Rules of the contract that the example already keeps and a replacement must:

- `Params` derives from `SelectorParams` (frozen, no unknown key, every float finite).
  Bound every field and give it a `description`. A randomised method takes a `seed`.
- **Pure**: the same input gives the same answer; no I/O but `data.scan.image(frame)`, no
  clock, no import of `smc.hardware`, `smc.testing`, `smc.ui` or `pymmcore_plus`
  (`tests/test_kit_imports.py` walks `src/smc/selection/` and fails on one). NumPy is
  there; any other library is a declared dependency in `pyproject.toml` in the same PR.
- `select` returns the fields of every well, **best first**: `route` gives a cell that
  two touching fields share to the first. A choice is a field centre in stage µm. Its
  `score` is larger-is-better on any scale; its `measures` are the numbers it decided
  on, keys in `snake_case` with their unit (`density_per_mm2`, `nn_median_um`,
  `doublet_fraction`), values finite.
- The layer **refuses** (never trims) a choice outside the well shrunk by
  `wall_margin_um`, on the excluded centre, a well with more than `fields_per_well`
  choices, two overlapping fields of one well, a non-finite number. Filter with
  `data.allowed(...)` and avoid overlap with `first_overlap` as the example does, and
  the refusal never fires.
- A selector that cannot use its input raises `SelectionError` with a sentence the
  owner can act on (`smc.selection.SelectionError`).
- A field chosen at another objective has that objective's footprint: use
  `data.matrix_um_per_px`, `data.image_shape_px`, `data.fov_um` and `data.cells_in`,
  never the scan's camera.
- Units in names (`_um`, `_px`); a pixel size of 0 is unknown, never defaulted.

## The test

Cells are placed by hand with `make_input`; the answer is worked out before the code
runs. `make_input` has one planned field per well, at the well's centre, and a camera
of `field_um` (100 by 100 µm by default), so a cell within 50 µm of the centre is in the
field and one further away is not.

```python
"""kebab-name on cells placed by hand: the answer is known before the code runs."""

from __future__ import annotations

from smc.selection import check_choices
from smc.selection.snake_name import ClassName, ClassNameParams
from smc.selection.testing import make_input


def test_the_field_that_holds_cells_is_chosen_and_scored_by_how_many() -> None:
    x_um, y_um = make_input([]).first_field("B2")  # the well's centre
    cells = [("B2", x_um + dx, y_um, "single") for dx in (-20.0, 0.0, 20.0)]
    data = make_input(cells)
    chosen = ClassName().select(data, ClassNameParams())
    assert [(c.well, c.x_um, c.y_um) for c in chosen] == [("B2", x_um, y_um)]
    assert chosen[0].score == 3.0
    assert chosen[0].measures == {"cells_in_field": 3.0}
    check_choices(chosen, data)  # what route checks on every choice


def test_a_field_with_no_wanted_cell_is_not_chosen() -> None:
    x_um, y_um = make_input([]).first_field("B2")
    far = [("B2", x_um + 400.0, y_um, "single")]  # outside the 100 um field
    unwanted = [("B2", x_um, y_um, "doublet")]  # in the field, not a wanted class
    for cells in (far, unwanted):
        assert ClassName().select(make_input(cells), ClassNameParams()) == []
```

Name each test for the behaviour or the failure it prevents. Add a case for every
decision the replacement makes (a tie, the wall margin, a well with no cells, a well
with more candidates than `fields_per_well`, a second objective's field with
`make_input(..., field_um=(...))`). A selector that counts doublets passes
`labels=("single", "doublet")` to `make_input`. A selector that reads pixels is tested
on a run folder the test writes: `make_input` has no frames.

## The entry point

Add the line under `[project.entry-points."smc.selectors"]` in `pyproject.toml`:

```toml
kebab-name = "smc.selection.snake_name:ClassName"
```

Entry-point metadata is written **at install time**: reinstall, or `route` says
`selector 'kebab-name' is not registered`. In a worktree run
`python scripts/dev/worktree.py setup`; in a plain clone `pip install -e ".[dev]"`.
`name` in the class equals the entry point's name; both are kebab-case.

## The page

`docs/algorithms/<name>.md`, one page per algorithm, with these four headings:

```markdown
# kebab-name

One sentence: what it chooses.

## What it measures

The numbers it decides on, how each is computed, the units, what is assumed.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_cells` | 1 | A field must hold at least this many cells of the wanted classes. |

## Tried on

The dataset (its name on the share), the stage that made the cells, the route's
parameters, the command that reproduced it.

## Numbers

What it chose on that dataset, with numbers: fields per well, the cells they hold,
what was left uncovered. Say what was not tried.
```

## Check it

Run, from the repository root (the worktree's `.venv` in a worktree):

```sh
ruff check src tests
mypy
pytest tests/test_selector_<snake_name>.py tests/test_kit_imports.py
```

Then try it on a dataset, on a copy (a tool that writes a stage changes the dataset's
`run.json`), as the guide says:

```sh
smc run route --offline --run-doc local/<copy>/run.json --param selector=kebab-name
smc dataset draw local/<copy>/run.json
```

Report which commands you ran and what they printed; do not claim a selector works on
a dataset you did not run it on.
