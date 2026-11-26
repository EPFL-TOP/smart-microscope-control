# Working on algorithms: a guide

For someone who has never seen this repository. You will write an algorithm that
looks at what a microscope scan acquired and decides which fields to image next, and
you will do it on your own computer, with no microscope and no Micro-Manager.

- [What this is for](#what-this-is-for)
- [Set up](#set-up)
- [Get a dataset](#get-a-dataset)
- [Load it in Python](#load-it-in-python)
- [Run the chain offline](#run-the-chain-offline)
- [Write a selector](#write-a-selector)
- [Test it](#test-it)
- [What never to assume](#what-never-to-assume)
- [What never to touch](#what-never-to-touch)
- [The issues and the pull request](#the-issues-and-the-pull-request)
- [What comes next](#what-comes-next)

## What this is for

smc is the software that controls the lab's microscopes. A **scan** images a plate:
one frame (an image) per field, each with a record of where the stage was, and a
detector then finds the cells in the frames. What to image next, for example at a
higher magnification, is a decision taken from that data. Two small algorithms make it:

- a **selector** chooses the fields (a well and a centre, in stage µm), best first;
- an **orderer** chooses the order the fields are visited in.

The measures you may decide on are the ones the scan allows: the density of cells, the
mean distance between cells, the proportion of doublets, and whatever else the frames
and the cells show.

The work is split between two people. **You** write the algorithm: data in, a decision
out, tested on synthetic data and looked at on a real dataset. **The owner** (the
repository's maintainer) connects the result to the tool that drives the microscope, in a second issue, and checks it at
the stand. What you write runs unchanged there, because the tool that drives the stand
(`route`) calls your selector by name, and the plate page lists the installed selectors
and orderers with their summaries and parameters, so that the owner chooses yours there
(`pip install -e .` once, so that its entry point is known, then restart the page).

The kit that makes this possible on a laptop:

- a **dataset**: a scan's folder (the frames and the cells), copied so that it needs no
  microscope;
- a **loader**, `smc.data`, that reads a dataset in Python;
- **offline runs**: `smc run NAME --offline` runs the tools that never drive the stand
  (`cells`, `density`, `route`, `report`, `nis-export`) on a copy of a dataset;
- the **selection contract**, `smc.selection`: what a selector is given and gives back,
  and the checks every choice goes through;
- a **drawing**, `smc dataset draw`: each well with its frames, its cells and the fields
  that were chosen, in order.

## Set up

You need Python 3.10 or later and git. You need **nothing of Micro-Manager**: no
adapters, no vendor software, no `smc doctor`. Ask the owner for access to the
repository (you work on branches of it, and never on `main`), then, in your clone:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

On Windows, in PowerShell, the first line is two commands, and the rest is the same:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

(If PowerShell refuses to run the script, `Set-ExecutionPolicy -Scope Process
RemoteSigned` allows it for this window only.) The Windows commands of this guide are
for PowerShell; at a `cmd.exe` prompt the activation is `.venv\Scripts\activate.bat`.
Clone the repository to a short path (for example `C:\src\smc`): the PyTorch install
below holds files with long names, and Windows refuses paths over 260 characters unless
long paths are enabled.

`pytest` skips the tests that need Micro-Manager's simulated microscope, with the reason
`Micro-Manager demo adapters not installed`. That is expected on your computer: those
tests run in the continuous integration on every pull request. **Never set the
environment variable `SMC_REQUIRE_MM` on your computer**: it turns those skips into
failures, and it is for the integration only.

Running the cell detection model needs PyTorch, which is a separate, large install:

```sh
pip install -e ".[dev,detect]"
```

Do that only if you re-detect cells yourself (see below); a dataset normally comes with
its cells already detected.

## Get a dataset

A dataset is a scan's folder: a run document (`run.json`), the frames as TIFF files, the
detected cells, and a `manifest.json` that lists every file with its size and its hash.
Datasets are never in the repository: they are zipped on the lab's share. They live in
the folder `<lab share>/smc-datasets/`: ask the owner for the path and for the name of
the dataset your issue names. The cell detection model's weights travel the same way.
[datasets.md](datasets.md) lists the datasets that exist: what each holds, how big it is,
what was checked on it and what was not. A dataset that has a `NOTES.md` beside its
`README.md` (the datasets of 2026-11-24 do) carries what the generator cannot know: read both.

Unzip it under `local/` of your clone (git ignores that folder):

```sh
mkdir -p local
unzip <lab share>/smc-datasets/<dataset>.zip -d local/
```

```powershell
mkdir local
Expand-Archive "<lab share>\smc-datasets\<dataset>.zip" -DestinationPath local
```

Check it before anything else. This reads every file the manifest lists and compares its
size and its hash, then loads the scan; it opens no microscope:

```sh
smc dataset check local/demo-dataset
```

```text
checked: 1/35
checked: 35/35
✓ 35 files match manifest.json (16.8 MB)
profile  demo-sample
frames   32 in wells B2, C3
cells    84 in cells/3be04c8f645e.json
```

(The outputs in this guide come from a small synthetic dataset, `demo-dataset`; yours
has its own name and more frames.) Exit code 0 means every file matched. Exit 1 names the
files that are missing or differ: copy the dataset again. Exit 2 means the folder is not
a dataset, or smc refuses the scan, and says why.

Then see what it holds:

```sh
smc dataset show local/demo-dataset/run.json
```

```text
run document   local/demo-dataset/run.json
profile        demo-sample
objective      Nikon 10X S Fluor
pixel size     1.0000 µm/px
image shape    512 x 512 px (height x width)
frames         32 (0 fields failed)
place in well  32 of 32 frames

well  frames  focus
B2    16      off 16
C3    16      off 16

cells          84, detected with blobs

well  single  doublet
B2    23      11
C3    39      11
```

**Keep the dataset as it came, and work on a copy.** Every tool that writes a result
(`cells`, `route`, `report`, ...) writes it into `run.json`, so the dataset would no
longer match its manifest and `smc dataset check` would report it changed. Copy the
folder once and run everything on the copy:

```sh
cp -r local/demo-dataset local/work
```

```powershell
Copy-Item -Recurse local\demo-dataset local\work
```

## Load it in Python

```python
from smc.data import cells_table, load_scan

scan = load_scan("local/demo-dataset/run.json")

print(scan.profile, scan.wells)
print(scan.pixel_size_um, scan.image_shape_px)

frame = scan.frames("B2")[0]
print(frame.name, frame.x_um, frame.y_um, frame.focus)
image = scan.image(frame)
print(image.shape, image.dtype)

if scan.cells is None:
    print(scan.cells_absent)
else:
    cells = scan.cells_of(well="B2")
    table = cells_table(cells)
    print(len(cells), table["label"][:3], table["x_um"][:3])
```

```text
demo-sample ('B2', 'C3')
1.0 (512, 512)
B2_0000 -41191.2 23191.2 off
(512, 512) uint16
34 ['single' 'single' 'doublet'] [-41284.7 -41251.7 -41005.2]
```

- `load_scan` reads the run document and refuses what cannot be used, with a message
  that names the file and the reason (`DatasetError`). It reads no frame yet.
- `scan.wells` is the plan's order; `scan.frames(well)` are the frames, each a record
  (`name`, `x_um`, `y_um`, `z_um`, `focus`, `well_dx_um`, `well_dy_um`, ...);
  `scan.image(frame)` is its pixels, a NumPy array of shape (height, width).
- `scan.cells` is the tuple of detected cells, or `None` when there are none:
  `scan.cells_absent` says why. A cell has a `well`, a `label` (its class: `single`,
  `doublet`, ...), a `score`, a position `x_um`, `y_um` in stage µm, the `frame` it was
  seen in (an index into `scan.result.frames`) and its box `box_px`.
- `cells_table(cells)` gives NumPy columns (`well`, `label`, `frame`, `score`, `x_um`,
  `y_um`, `row_px`, `col_px`, ...), one entry per cell. `pandas.DataFrame(cells_table(cells))`
  is a table; pandas is not a dependency, install it yourself if you want it.
- The camera is `scan.matrix_um_per_px` (it turns a pixel offset from a frame's centre
  into a stage offset) and `scan.pixel_size_um`.

The `README.md` inside every dataset explains its layout, its units and its conventions.

## Run the chain offline

The tools that read a scan and never drive the microscope run on your copy with
`--offline`. No stand is opened; the profile is the one the run document recorded; the
record of each stage says it was made offline. A tool that drives the stand is refused.

```sh
smc run cells --offline --run-doc local/work/run.json --param detector=blobs
```

```text
offline: no stand is opened; the profile, demo-sample, is the run document's
cells: 1/32 B2 B2_0000
cells: 32/32 C3 C3_0015
✓ cells done in 0.2 s — local/work/run.json, stage cells
```

`cells` re-detects the cells. On a real dataset omit `detector=blobs`: it uses the
trained model, which needs `pip install -e ".[dev,detect]"` and the weights in
`local/models/` (ask the owner for them). `detector=blobs` counts bright round spots on
the synthetic plate of the demo profile and **means nothing on a real sample**. Skip this
step when the dataset's cells are the ones you want.

```sh
smc run density --offline --run-doc local/work/run.json
```

```text
offline: no stand is opened; the profile, demo-sample, is the run document's
density: 1/2 B2
density: 2/2 C3
✓ density done in 0.0 s — local/work/run.json, stage density
```

`density` counts the cells of each well and class and divides by the area that was
imaged. Next, `route` chooses the fields and the order to visit them in. With no
`selector` parameter it uses the rule the project has always used (`most-cells`: the
fields that hold the most cells, then the shortest path), counting the `single` class:

```sh
smc run route --offline --run-doc local/work/run.json --param fields_per_well=4
```

```text
offline: no stand is opened; the profile, demo-sample, is the run document's
route: 1/2 B2
route: 2/2 C3
✓ route done in 0.0 s — local/work/run.json, stage route
```

The result is printed after these lines as JSON: the stops, in visiting order, each with
its well, its position and the cells it holds. Your selector replaces the rule by name
(`selector=<name>`), and an orderer is chosen the same way (`orderer=<name>`).
[`docs/plugins/route.md`](../plugins/route.md) lists `route`'s parameters and what it
checks.

A parameter that is a list or an object is **JSON**: the classes to count
(`labels=["single","doublet"]`) and your selector's own parameters (`selector_params`).
In a POSIX shell, single quotes keep the double quotes of the JSON:

```sh
smc run route --offline --run-doc local/work/run.json --param selector=<name> --param 'selector_params={"min_cells":3}'
```

Windows shells quote it in their own ways (PowerShell 5.1, PowerShell 7 and `cmd.exe`
differ, and none is tried here), so give JSON through a **preset file**, which no shell
reads. Save this as `local/my-route.toml`; each item of the list is a `--param`:

```toml
[params]
route = ['labels=["single", "doublet"]', 'selector_params={"min_cells": 3}']
```

```sh
smc run route --offline --run-doc local/work/run.json --preset local/my-route.toml --param selector=<name>
smc run route --offline --run-doc local/work/run.json --preset local/my-route.toml --param selector=<name> --param orderer=as-chosen
```

A `--param` replaces the preset's value of the same key. `as-chosen` keeps your
selector's own order. A mistake of yours is a refusal and never a silent change: the
stage ends `failed` with a sentence that starts `selector <name>:` and names the choice.

Draw what was chosen. One picture per well, in `local/work/draw/` (the pictures it
writes are replaced; nothing else is touched):

```sh
smc dataset draw local/work/run.json
```

```text
route  route (selector <name>, orderer as-chosen), 7 stops
cells  84 detected
drawing: 1/2
drawing: 2/2
B2  16 of 16 frames  4 stops  34 cells  local/work/draw/B2.png
C3  16 of 16 frames  3 stops  50 cells  local/work/draw/C3.png
✓ 2 pictures written to local/work/draw
```

Each picture shows the well's frames, the wall, the cells coloured by class, and the
chosen fields numbered in the order they are visited, with the selector's score in
brackets and the path between them. Look at it with the owner: it is how a choice is
judged. `--wells "B2,C3"` draws some wells only (quote the list in PowerShell) and
`--max-px` sets the size of the pictures.

Last, the report and the position list for NIS-Elements, the software the lab acquires
with:

```sh
smc run report --offline --run-doc local/work/run.json
smc run nis-export --offline --run-doc local/work/run.json
```

```text
offline: no stand is opened; the profile, demo-sample, is the run document's
✓ report done in 0.0 s — local/work/run.json, stage report
offline: no stand is opened; the profile, demo-sample, is the run document's
✓ nis-export done in 0.0 s — local/work/run.json, stage nis-export
```

`report` writes one HTML file (the plate's density, the cells, the route) and
`nis-export` writes the route as an XY position list that NIS-Elements loads. Both read
the route: run `route` again and smc tells you which of them is out of date
(`! nis-export read route from an earlier run; run it again`).

## Write a selector

In Claude Code, `/new-selector <name>` writes everything below for you, with a working
example to replace (`<name>` in kebab-case, for example `dense-fields`). By hand, a
selector is four things:

1. **The module**, `src/smc/selection/<snake_name>.py`: a `Params` class (it derives
   from `SelectorParams`: frozen, no unknown key, every number finite) and a class with
   `name`, `summary`, `Params` and `select(data, params)`.
2. **The entry point**, a line in `pyproject.toml` under
   `[project.entry-points."smc.selectors"]`: `<name> = "smc.selection.<snake_name>:<ClassName>"`.
   Entry points are written when smc is installed: run `pip install -e ".[dev]"` again,
   or `route` says the selector is not registered.
3. **The test**, `tests/test_selector_<snake_name>.py`.
4. **The page**, `docs/algorithms/<name>.md`: what it measures, its parameters, what was
   tried on which dataset, with numbers.

`select(data, params)` is given a `SelectionInput` and returns the chosen fields of
every well, best first:

- `data.scan`: the scan as loaded above (`wells`, `frames`, `image`, `cells`, `plan`);
- `data.cells`: the cells (refuses, with the reason, when the scan has none);
- `data.labels`, `data.fields_per_well`, `data.wall_margin_um`,
  `data.exclude_centre_um`: what `route` was asked for;
- `data.matrix_um_per_px`, `data.image_shape_px`, `data.fov_um`: the **imaging
  objective's field**, which is the scan's unless `route` was asked to image with
  another objective;
- `data.allowed(x_um, y_um, well)`: whether a field centred there keeps to the well's
  margins; `data.cells_in(x_um, y_um, well)`: the cells that field holds;
  `data.first_field(well)`: the scan plan's first field of the well.

A choice is `Choice(well, x_um, y_um, score=None, measures={})`: a field centre in stage
µm, a `score` of your own (larger is better, on any scale) and the `measures` you decided
on, with their unit in the name (`density_per_mm2`, `nn_median_um`, `doublet_fraction`),
finite numbers. `route` writes them next to the stop and the drawing shows the score.

The rules of the contract:

- **Pure.** The same input gives the same answer: no I/O but `data.scan.image(frame)`, no
  clock, no import of the hardware layer. A randomised method takes a `seed` parameter.
  NumPy is available; any other library is a dependency you declare in `pyproject.toml`
  in the same pull request.
- **The layer checks every choice and refuses a bad one** (the stage then ends `failed`
  and names the choice): a well the scan does not have, a number that is not finite, a
  field that leaves the well shrunk by `wall_margin_um` or touches the excluded centre,
  more than `fields_per_well` choices in a well, two overlapping fields of one well.
  `data.allowed` and `smc.plugins.route.algorithm.first_overlap` keep you inside.
- A selector that cannot use its input raises `SelectionError` with a sentence the owner
  can act on.

An orderer is the same with `order(choices, data, params)`, which returns a permutation
of `range(len(choices))`, and the entry-point group `smc.orderers`. The full contract is
in [`docs/design/algorithms.md`](../design/algorithms.md) §5.

## Test it

A selector is tested twice: on cells placed by hand, whose answer you know, then on a
real dataset.

`smc.selection.testing.make_input` builds a `SelectionInput` in memory, with no file and
no microscope, from cells given as `(well, x_um, y_um, label)`:

```python
from smc.selection.testing import make_input

data = make_input(
    [("B2", 10.0, 0.0, "single"), ("B2", 12.0, 0.0, "doublet")],
    labels=("single", "doublet"),
    field_um=(100.0, 100.0),
)
```

Its plate has A1 at the origin, the plan has one field per well, at the well's centre,
and the camera is upright and not mirrored. It has **no frames**: a selector that reads
pixels is tested on a run folder the test writes. Name each test for the behaviour or
the failure it prevents (`test_a_field_on_the_wall_margin_is_not_chosen`), and add a case
for every decision your algorithm makes.

Then run it on the dataset (the commands of *Run the chain offline*), draw it, and look
at the pictures with the owner. Write what you found, with numbers, on the algorithm's
page.

Before every push, the checks the continuous integration runs:

```sh
ruff check .
ruff format --check .
mypy
pytest
```

## What never to assume

- **A stage position is a readback.** A frame's `x_um` and `y_um` are where the stage
  said it was after the move, which is where the frame's centre is. They are not the
  target, and they are not rounded to it.
- **A pixel size of 0 is unknown.** smc reports it and never defaults it; so must you.
- **A frame's `focus` may be `unlocked`.** `locked` means the autofocus held the focus;
  `unlocked` that it was used and did not hold; `off` that the stand has one and the scan
  did not use it; `none` that the stand has none. A frame that is not `locked` may be
  out of focus: say what your algorithm does about it.
- **`well_dx_um` and `well_dy_um` may be `None`.** A scan written before smc recorded
  them has none: unknown, never zero.
- **A field chosen at another objective has that objective's footprint.** Choose with
  `data.matrix_um_per_px`, `data.image_shape_px` and `data.fov_um`, never with the
  scan's camera: a field chosen at 40× from a 10× scan is about four times smaller on
  each side.
- **The cells are the detector's, not the truth.** Look at `scan.cells_result`: the model
  that found them, its score threshold, its classes.
- **Read the fields of `SelectionInput` and `Scan` only.** Reading another stage's record
  through `scan.document` is a read smc does not see: the result would not be marked out
  of date when that stage is run again.

## What never to touch

An algorithm change lives in `src/smc/selection/`, `tests/`, `docs/algorithms/` and
`pyproject.toml` (its entry point, and a dependency if it needs one). It does not touch:

- `src/smc/hardware/`, `profiles/` or `src/smc/ui/`: how smc reaches a microscope, what
  each microscope is, and the plate page are the owner's;
- `docs/adr/` and `docs/design/` (decisions and designs), `.github/`, `.claude/` and
  `CLAUDE.md` (how the project works): the owner's too. A pull request that touches one
  of these paths, or `src/smc/hardware/` or `profiles/`, needs the owner's review before
  it can merge (`.github/CODEOWNERS`).

Never import `pymmcore_plus` or `pymmcore`, never run `pytest -m hardware`, and never run
anything that moves a stage: your work needs no microscope, and a test checks that
`smc.data` and `smc.selection` import none of it.

## The issues and the pull request

Each result is two issues. **Yours** is an *Algorithm* issue (`area: algorithms`, from
the template: the goal, the data in, the decision out, how it is judged). It is done when
the algorithm passes its tests in the continuous integration and you have looked at it on
a dataset with the owner. **The owner's** connects the result to the tool that drives the
microscope and carries the check at the stand; it names yours as its dependency.

In Claude Code, `/develop <issue>` implements an issue that is labelled `status: ready`:
it plans from the issue, writes the test first, has cheap reviewer agents attack the
change, and opens the pull request. `/feature <issue>` does the same for a small one.
Without Claude Code: branch from `main` as `feat/<issue>-<slug>`, make the checks above
pass, and open a pull request titled in the Conventional Commits form
(`feat(algorithms): a selector by density`), with `Closes #<issue>` in its body and the
template filled honestly, including what you did not verify. **Agents never merge.**
Under ADR-0010 the lead of the analysis track merges a pull request whose paths are all
the analysis track's, once `/review` says `ready to merge` and the light set is green;
`.github/CODEOWNERS` is the map of the paths, and `python scripts/dev/tracks.py <path> ...`
tells the track of one. A pull request that touches a stand path (a shared file, the
selection contract `smc.selection.api`, `registry.py`) needs the owner's approval on
GitHub first, and that need is an issue for the stand track before it is an edit: a need
across the border is never an edit.

## What comes next

Two pieces are filed apart. The **acquisition loop**, which goes from position to
position and images each, is built on the simulated microscope with a synthetic sample, so
it can be taken by whoever has Micro-Manager's demo adapters (`mmcore install
--test-adapters`); its first run at a real microscope is the owner's. **Decisions taken
during an acquisition**, rather than after a scan, are not covered by this contract.
