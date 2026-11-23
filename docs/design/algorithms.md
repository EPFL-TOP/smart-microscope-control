# Algorithms design — a collaborator's kit for selection algorithms on acquired data

- **Status**: design for issue #554 (2026-11-18, with the owner), built by
  #564 (§3), #565 (§2), #566 (§4), #567 (§5), #568 (§6) and #554 itself (§7);
  the first task is #570, its wiring #569.
- **Owner**: the design session. **Executors**: `/develop` sessions, one per
  issue, the collaborator's included.
- **Rule**: as in M1 to M5, this document is the contract between issues built
  in parallel. Names, module paths and signatures below are fixed; an executor
  who needs to change one stops and comments on its issue (`## Plan
  deviation`). M1's conventions and CLI rules, and M3's plugin API (§1), hold
  everywhere this document does not say otherwise; §4 and §5 amend m3 §1.10,
  §3.5 and §11, and say so where they do.

Read first: [ADR-0004](../adr/0004-plugin-architecture.md) (the pure algorithm
and the hardware loop are separate) and m3 §3.4, §3.5, §9 and §11.

## 0. Scope and the split

A second developer writes algorithms that run on what a scan acquired (the
frames, their records, the detected cells) and decide which fields to image
and in what order. They work on a laptop with no microscope and no
Micro-Manager, and what they write runs online unchanged, through `route`,
when the owner connects it at the stand. This is ADR-0004's split made into
an arrangement between two people:

- **The collaborator** owns the *pure algorithm*: data in, a decision out,
  unit-tested on synthetic data and checked on a real dataset. Their issues
  carry `area: algorithms` and touch nothing of `smc.hardware`, `profiles/`
  or `smc.ui` (§7).
- **The owner** owns the *hardware loop*, the page and the stand: a second
  issue connects each result and carries the check at a stand (§7.2).

The kit is: a dataset exported from a run folder (§2), a loader with no
hardware (§3), the plugins that never drive the stand run offline (§4), the
contract of a selection algorithm (§5), a drawing of what it chose (§6), and
the way two people work in one repository (§7).

**Decided with the owner (2026-11-18)**: the collaborator has write access
to the repository and works with Claude Code and this repository's skills;
datasets travel as a zip on the lab's share; `main` asks for the code
owner's review on the paths of §7.4 only; tools run offline through `smc
run NAME --offline` rather than a second set of commands; the contract is
two halves, a selector and an orderer, chosen by name; the algorithms live
in this repository under `src/smc/selection/`; the first task (#570) uses
the 10× scan with cells of 2026-11-19 and a 40× scan of the same wells taken
the same day.

Out of scope: the acquisition loop (#556, its engine is an ADR), decisions
taken *during* an acquisition (#32), the page's choice of a selector (#569: the
page's half is fixed in [m5 §21](m5-web-ui.md)).

## 1. Measured (2026-11-18)

On the main checkout's `.venv` (Python 3.13, pymmcore-plus as pinned):

- `import smc.run`, `smc.plugins.slots`, `smc.plugins.scan`,
  `smc.plugins.cells`, `smc.plugins.route`, `smc.plugins.api`,
  `smc.hardware.profile`, `smc.plates` and `smc.cli` leave `pymmcore_plus`
  out of `sys.modules`. `smc.testing` imports it (`fakes.py` imports
  `DeviceType`), so nothing of the kit may import `smc.testing` outside
  tests. `smc.plugins.slots` imports `smc.hardware.microscope` (not
  `pymmcore_plus`).
- `Profile.load("nikon-ti2")` loads with no Micro-Manager. Its
  `[camera.pixel_size_um]` holds the 4×, 10× and 40× sizes (2.73909,
  1.12447, 0.276136 µm); a profile's `[microscope] name` equals its file stem
  for the five profiles of `profiles/`.
- The run document of the 10× scan of 2026-11-17 (`local/`, private): stages
  `goto-well`, `camera-calibration`, `plate-walls`, `scan-plan`, `scan`,
  `timing`, all `done` on `nikon-ti2`, none a dry run, no path in any
  string parameter; the scan holds 218 frames of 1412 × 1412 px at
  1.1249 µm/px with `2-Plan Fluor 10x`, 1.1 GB of TIFFs; each frame's record
  has `well`, `index`, `name`, `file` (`scan/<token>/<well>/<name>.tif`),
  `x_um`, `y_um`, `z_um`, `focus`, `pfs_offset`, `t_s`, `well_dx_um`,
  `well_dy_um`. The plate held beads, so it has no `cells` record.
- The runner (`smc/plugins/runner.py`) reads only `microscope.profile` (its
  `[microscope] name`, into the record's `profile`) and `microscope.dry_run`,
  and calls `microscope.has` for each capability in `requires`. `cells`,
  `density` and `nis-export` never use their `microscope` argument; `report`
  reads its profile's name; `route` reads the profile's autofocus factors and
  limits, and asks the turret for its labels and pixel sizes only when
  `objective` names another objective than the scan's
  (`route/plugin.py` `_imaging_with`).
- `RunDocument.unusable` refuses an upstream whose `profile` differs from
  the reader's, and a dry-run upstream read by a real run.
- The turret's `pixel_size_um(label)` is the profile's size divided by the
  core's magnification factor, the same factor for every label
  (`backends/mm.py`), so the ratio of two objectives' sizes is the ratio of
  the profile's entries.
- `scripts/dev/select_fields.py` draws with Pillow (`render_png`,
  `canvas_of`, `place_frame`, `stretch_to_uint8`); Pillow (≥ 10.1) and SciPy
  come only through scikit-image, and nothing in `src/` imports them.
- `smc run` always opens a `Microscope` (`Microscope.open`, which needs the
  adapters), even with `--dry-run`.

## 2. The dataset (#565)

**A dataset is a run folder.** It keeps the run document and the files the
scan and the cells wrote at the same relative paths, so every tool that
reads a run folder reads a dataset, and a dataset loads with the same
`load_scan` (§3).

### 2.1 `smc dataset export RUN_DOC --out DIR`

`src/smc/data/dataset.py` (`export_dataset`), the command in
`src/smc/data/cli.py`.

```python
def export_dataset(
    run_doc: Path,
    out: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> DatasetManifest: ...
```

- **Refused before anything is written** (exit 2, a `DatasetError` naming
  the reason), in this order: `out` exists and is a file, or is a folder
  that is not empty (the export never overwrites and never deletes); the
  `scan` slot has no provider, or it is not usable (`RunDocument.unusable("scan",
  profile=<its own profile>, dry_run=False)`), or it was a dry run (the
  refusals of `load_scan`, §3); one or more frame files are missing (their
  count and the first five paths); on Windows, a destination path of 260
  characters or more (long paths are not assumed); and, when the free space
  of `out`'s disk can be read, less than the files to copy need. The scrubbed
  document (§2.2) is built, and validated, before the first file is
  written. `out`'s missing parent folders are created.
- **Written**, in this order: every frame of the scan's result at its own
  relative path; the `cells` file when `load_scan` found cells (the `cells`
  slot's provider is `done`, not stale, on the scan's profile; else no cells,
  and the manifest says why); `run.json`, the scrubbed run document (§2.2),
  whatever the source's file name; `README.md` (§2.3); and `manifest.json`
  last, so a folder with no manifest is an interrupted export (FM-106). A
  frame is copied in 1 MiB chunks and hashed as it is read: the manifest
  holds the hash of the **source's** bytes, so a bad write is found by
  `check` and not recorded as the truth (hashing the copy afterwards would
  bake it in, and read the file once more). A file that already exists is
  never opened for writing.
- **Kept and not copied**: every other stage record stays in `run.json`
  (staleness reads them), but no other file is copied; the README says so.
- **Exit codes**: 0 written; 1 an I/O error while writing (the partial
  folder stays, with no manifest, and the message says how many files were
  written, and that the folder is to be removed); 2 refused.
- **Progress**: the function takes a callback `(done, total)` called after
  each file copied. The CLI prints a line at most once a second, the first
  and the last always, as `smc run` does: a `rich` bar on a redirected
  stream draws nothing and leaves an empty line (measured, rich 15).

### 2.2 What is scrubbed

Host and account names reach a run document only through paths (a model
path, an output file, an error that quotes a file). In every record's
`params`, `result`, `checkpoint` and `error`, recursively, in string values
(a key is never rewritten):

- a string that **starts like** an absolute path (POSIX `/…`, Windows `X:\…`
  or `X:/…`, UNC `\\…`) becomes its last component (split on `/` and `\`);
- inside any other string, each substring matching one of the three
  patterns below becomes its last component;
- a path with no component below its root, its drive, its UNC share or its
  account folder (`/`, `C:\`, `\\host\share`, `/Users/<name>`,
  `/home/<name>`, `X:\Users\<name>`) becomes `~`: its last component would be
  the account;
- each rewritten value's JSON pointer (`/stages/cells/params/model_path`,
  RFC 6901: `~` is `~0` and `/` is `~1` in a key) goes into the manifest's
  `scrubbed`, in the document's order: the stages as the file has them, in a
  record `params`, `result`, `checkpoint`, `error`, each depth first.

```text
(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s'"]*     a drive, not the end of a word
\\\\[^\s'"]*                               a UNC path
(?<![\w.~:/-])/[^\s'"/]+/[^\s'"]*          a POSIX path of two components or more
```

Why these three and not `[A-Za-z]:[\\/]` with `/(?:Users|home)/`: a drive
pattern with no word boundary rewrites the end of any word before `:/`
(`file:///Users/alice/x.tif` became `filx.tif`), and `/(?:Users|home)/` leaves
`/mnt/lab/alice/run1` alone inside a sentence, where the `error` of a failed
`cells` quotes the run folder (`load_cells`); both measured on 2026-11-18,
planning #565. Known gaps, left as they are: a `file:///` URL, and a
single-component path such as `/media/<name>` inside a sentence. The rule is
idempotent: exporting a dataset rewrites nothing (`scrubbed` is empty). On the
real 10× run document of 2026-11-17 it rewrites no string (§1: it holds no
path).

Relative paths (`scan/<token>/A1/A1_0003.tif`) are never touched. The TIFFs
are copied unchanged: their description is the frame's record (m3 §8, the
scan's `ScanFrame`), which holds no path but the relative one.

### 2.3 The manifest and the README

```python
class DatasetFile(BaseModel):
    bytes: int
    sha256: str


class DatasetManifest(BaseModel):
    schema_version: int = 1
    smc_version: str
    exported_utc: AwareDatetime
    source: str  # the source run document's file name, never its path
    profile: str  # the scan record's profile ([microscope] name)
    objective: str | None
    pixel_size_um: float
    image_shape_px: tuple[int, int]
    wells: tuple[str, ...]  # the plan's order
    frames: int
    cells: bool
    cells_absent: str | None  # why there are no cells, when there are none
    files: dict[str, DatasetFile]  # POSIX relative path -> size and hash
    scrubbed: tuple[str, ...]  # JSON pointers rewritten by §2.2
```

Both models are `extra="forbid"`, `frozen=True`; `files` holds every file
but `manifest.json`, in the order written (frames in the scan's order, the
cells file, `run.json`, `README.md`). `run.json` and `manifest.json` are
written as the run document is (`write_stage`'s dump: ASCII, indent 2, a
final newline), each through `write_atomically` (FM-38). The generated
`README.md` (English, plain, written before the manifest, so it lists no
hash) says: what the scan is (date, profile, objective, pixel size, image
shape, wells, frames, cells or why not); the layout; the units and
conventions (stage µm readbacks, `well_dx_um`/`well_dy_um` as the readback
minus the well's centre in stage axes, `(dx, dy) = M @ (d_col, d_row)` for the
matrix, `(d_col, d_row)` being the pixel offset from the frame's centre as in
`smc.plugins.geometry`, a pixel size of 0 is unknown); how to load it (`from
smc.data import load_scan`); how to check it (`smc dataset check`, and that a
tool that writes a stage into the folder's `run.json` makes `check` report it
changed: work on a copy); and that only the scan's frames and the cells file
were copied. Its fixed text is ASCII (`um`, no `µ`), so that any editor reads
it; the objective's name is the one non-ASCII word it can hold. Neither file
names the source or the destination folder (FM-50).

### 2.4 `smc dataset check DIR`

`src/smc/data/dataset.py` (`check_dataset`). Reads `DIR/manifest.json` (a
missing one says this is not a dataset, or an export that did not finish),
validates it (`utf-8-sig`, a `schema_version` above 1 is a newer smc's), and
refuses a listed path that is absolute or leaves the folder before it reads
any file; checks every listed file's size and SHA-256, each read once (FM-39);
then, only when every file matched, loads `DIR/run.json` with `load_scan`.
Files of the folder that the manifest does not list are ignored: the
collaborator's own outputs (§6's drawings) live beside the dataset. Exit 0
when all hold; 1 when a file is missing or differs (each named, the first 20,
then a count); 2 with no readable manifest, with a manifest that lists a path
outside the folder, when `load_scan` refuses, and for an exception `check`
did not expect: an unexpected exit 1 would read as a verdict, "a file differs"
(FM-82).

### 2.5 How a dataset travels

Zipped onto the lab's share (the owner names the folder in the guide,
§7.6), never in the repository. The collaborator unzips it under `local/`
in their clone (ignored by git, `.gitignore`'s `/local/`) and runs `smc
dataset check` first. The cell model's weights travel the same way.

## 3. The loader (#564)

Package `src/smc/data/`: `__init__.py` re-exports `load_scan`, `Scan`,
`cells_table`, `DatasetError`; `scan.py` holds them. Nothing in
`src/smc/data/` imports `smc.hardware`, `smc.testing` or `smc.ui` (§8).

```python
class DatasetError(ValueError):
    """A run folder or dataset that cannot be read; the message names the file and why."""


def load_scan(run_doc: Path | str) -> Scan: ...
```

```python
@dataclass(frozen=True, slots=True)
class Scan:
    run_dir: Path
    document: RunDocument
    profile: str
    plan: ScanPlanResult
    result: ScanResult
    wells: tuple[str, ...]
    geometry: Mapping[str, WellGeometry]
    cells: tuple[Cell, ...] | None
    cells_result: CellsResult | None
    cells_absent: str | None

    @classmethod
    def from_results(
        cls,
        run_dir: Path,
        document: RunDocument,
        *,
        plan: ScanPlanResult,
        result: ScanResult,
        cells_result: CellsResult | None,
        cells: Sequence[Cell] | None,
    ) -> Scan: ...

    @property
    def pixel_size_um(self) -> float: ...
    @property
    def matrix_um_per_px(self) -> Matrix2: ...
    @property
    def image_shape_px(self) -> tuple[int, int]: ...
    def frames(self, well: str | None = None) -> tuple[ScanFrame, ...]: ...
    def image(self, frame: ScanFrame) -> NDArray[Any]: ...
    def cells_of(
        self, *, well: str | None = None, frame: int | None = None
    ) -> tuple[Cell, ...]: ...
```

- **`load_scan`** loads the document (`RunDocument.load`; its
  `RunDocumentError` becomes a `DatasetError`), takes the `scan` slot's
  provider, refuses it as §2.1 does (no provider, not usable on its own
  profile, a dry run), and takes the `scan-plan` provider the same way. The
  cells are the `cells` provider's when it is usable on the scan's profile
  (`load_cells`); otherwise `cells` and `cells_result` are `None` and
  `cells_absent` holds the reason (`unusable`'s text, or "no cells record").
  Frames are not read at load.
- **`wells`** is the plan's order; **`geometry`** is `well_geometry(plan,
  well)` for each. The camera is the scan's (`result.matrix_um_per_px`,
  `result.image_shape_px`), never the plan's copy (FM-124).
- **`image(frame)`** is `smc.plugins.slots.load_frame(run_dir, frame)` (a
  function; `ScanResult` has no such method); its `OSError`, `ValueError` and
  `tifffile.TiffFileError` (named, not a `ValueError` before tifffile
  2026.x) become a `DatasetError` naming the frame.
- **`from_results`** builds a `Scan` from results already read (`route`
  uses it, §5.6); it checks nothing `ctx.upstream` has checked.
- `cells_of(well=…)` and `cells_of(frame=…)` filter by the cell's `well` and
  `frame` (the index into `result.frames`); both may be given.

```python
def cells_table(cells: Sequence[Cell]) -> dict[str, NDArray[Any]]: ...
```

Columns, one entry per cell, in the input's order: `well` and `label`
(`str` arrays), `frame` (`int64`), `score`, `x_um`, `y_um`, `row_px`,
`col_px` (the box's centre), `row0_px`, `col0_px`, `row1_px`, `col1_px`
(`float64`). `pandas.DataFrame(cells_table(cells))` is a table; pandas is
not a dependency.

**`smc dataset show RUN_DOC`** prints, through `rich`: the profile, the
objective, the pixel size and image shape, each well with its frame count
and its frames per focus state, and the cells per class per well, or why
there are none. Exit 0, or 2 when `load_scan` refuses. The `smc dataset`
group is a `typer` sub-app in `src/smc/data/cli.py`, added to
`src/smc/cli.py` by #564; #565 and #568 add their commands to the
sub-app only.

## 4. Offline runs (#566)

Amends m3 §1.10 (the CLI) and §1.3 (an optional plugin attribute).

### 4.1 `smc run NAME --offline --run-doc RUN_DOC`

- **Which plugins**: those whose class sets `offline: ClassVar[bool] =
  True`: `cells`, `density`, `route`, `report`, `nis-export`. The attribute
  is optional (`getattr(plugin, "offline", False)`); the registry does not
  require it, so a third-party plugin may set it. Any other is refused
  (exit 2): "NAME drives the stand; --offline runs: cells, density,
  nis-export, report, route" (the list from the registry, sorted).
- **The profile** is the one the run document recorded: the `profile` of
  the providers of the plugin's `reads` that exist. They must agree (they
  do whenever `unusable` passes). An explicit `--profile` (or
  `SMC_PROFILE`) that differs is refused naming both; none is the recorded
  one. No provider of any read slot: refused, "nothing upstream: an
  offline run reads a run document made at a stand". The profile is loaded
  by name from the search paths (`Profile.load`), which needs no
  Micro-Manager (§1).
- `--offline` with `--dry-run` is refused (usage error, exit 2).
  `--resume`, `--param` and `--preset` behave as on a stand.
- No stand is opened: the plugin runs on `Microscope.offline(profile)`
  (§4.2). The record carries `profile` = the profile's name, `dry_run` =
  false, and `offline` = true (§4.3).

### 4.2 `Microscope.offline(profile)`

`src/smc/hardware/microscope.py`; the owner's path (§7.4).

```python
class Microscope:
    @classmethod
    def offline(cls, profile: Profile) -> Microscope: ...
    @property
    def is_offline(self) -> bool: ...
```

(A class method and a property cannot share a name: the constructor is
`offline`, the property `is_offline`.)

- A facade with **no core and no device**: `has(capability)` is false for
  every capability, `get` returns `None`, `require` raises
  `CapabilityMissingError` whose message says that no stand is open (offline
  run). Every move therefore fails before it starts, by construction: there
  is no capability to move with.
- `profile` is the profile given; `dry_run` is false; `is_offline` is true, and false on every other `Microscope`.
- Creating one never imports `pymmcore_plus` and never calls `open_core`.
  Whatever the facade's other members do with no core (a state, a stop, a
  close) is a no-op or an empty answer; the plan of #566 lists them.

### 4.3 The stage record's `offline`

`StageRecord.offline: bool = False`, omitted from the JSON when false (as
`preset` is), so every existing run document reads and writes unchanged
and `SCHEMA_VERSION` stays 1. An offline run's record says it was made off
the stand; nothing else changes (staleness, `unusable`).

### 4.4 `route` offline

`_imaging_with` on an offline microscope (`microscope.is_offline`), with `objective` naming another
objective than the scan's: the labels are the keys of the profile's
`[camera] pixel_size_um` and the sizes its entries. The ratio is the one the
stand would give (§1: the magnification factor divides both), so `assumed`
says what it already says ("the field of <objective> is the scan's scaled
by the profile's pixel sizes") and nothing more. An objective absent from
the map is the same refusal as on a stand, naming the map. `_offset_profile`
reads the profile, as now.

## 5. The selection contract (#567)

Amends m3 §11 (route chooses through a selector and an orderer) and §3.5
(`RoutePosition` gains two optional fields).

### 5.1 The package

`src/smc/selection/`:

- `api.py`: `SelectorParams`, `OrdererParams`, `SelectionInput`, `Choice`,
  `Selector`, `Orderer`, `SelectionError`;
- `registry.py`: the two entry-point groups;
- `checks.py`: what the layer checks (§5.5);
- `builtin.py`: `MostCells`, `Nearest`, `AsChosen` (§5.4);
- `testing.py`: `make_input` for tests (§5.7);
- one module per algorithm, `src/smc/selection/<name>.py` (a package
  `src/smc/selection/<name>/` when it grows), the collaborator's.

Nothing in `src/smc/selection/` imports `smc.hardware`, `smc.testing` or
`smc.ui` (§8). It may import `smc.data`, `smc.plugins.slots`,
`smc.plugins.geometry` and `smc.plugins.route.algorithm`; `route`'s plugin
imports `smc.selection`, and `smc.selection` never imports
`smc.plugins.route.plugin`.

### 5.2 Inputs and outputs

```python
class SelectorParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class OrdererParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SelectionError(ValueError):
    """Data a selector or an orderer cannot use, or a choice the layer refuses."""
```

```python
@dataclass(frozen=True, slots=True)
class SelectionInput:
    scan: Scan
    labels: tuple[str, ...]
    fields_per_well: int
    wall_margin_um: float
    exclude_centre_um: float
    objective: str | None
    matrix_um_per_px: Matrix2
    image_shape_px: tuple[int, int]

    @property
    def fov_um(self) -> tuple[float, float]: ...
    @property
    def cells(self) -> tuple[Cell, ...]: ...
    def allowed(self, x_um: float, y_um: float, well: str) -> bool: ...
    def cells_in(
        self,
        x_um: float,
        y_um: float,
        well: str,
        *,
        labels: Sequence[str] | None = None,
    ) -> tuple[Cell, ...]: ...
    def first_field(self, well: str) -> tuple[float, float]: ...
```

- `scan` holds cells (route reads the `cells` slot); `cells` returns them
  and raises `SelectionError` if `scan.cells` is `None`.
- `matrix_um_per_px` and `image_shape_px` are the **imaging objective's**
  field (the scan's, or scaled per m3 §11), not the scan's: a field chosen
  at 40× from a 10× scan has the 40× footprint. `fov_um` is `fov_of` of
  them.
- `allowed` is `route.algorithm.allowed` with this input's matrix, shape,
  margin and excluded centre. `cells_in` is route's counting arithmetic (a
  cell is in a field when `pixel_of_stage` of its centre falls in the
  footprint, m3 §11), filtered to `labels` (default: this input's), made
  public by #567 as `route.algorithm.cells_in_field(cells, x_um, y_um, *,
  matrix, image_shape_px) -> tuple[Cell, ...]`. `first_field` is the plan's
  first field of the well, the reference route's order starts from.

```python
@dataclass(frozen=True, slots=True)
class Choice:
    well: str
    x_um: float
    y_um: float
    score: float | None = None
    measures: Mapping[str, float] = field(default_factory=dict)
```

A choice is a field centre in stage µm. `score` is the selector's own
(larger is better, any scale); `measures` are the numbers it decided on,
keys in `snake_case` with their unit (`density_per_mm2`, `nn_median_um`,
`doublet_fraction`), values finite.

### 5.3 The two protocols

```python
class Selector(Protocol[SP]):
    name: str
    summary: str
    Params: type[SP]

    def select(self, data: SelectionInput, params: SP) -> Sequence[Choice]: ...
```

```python
class Orderer(Protocol[OP]):
    name: str
    summary: str
    Params: type[OP]

    def order(
        self, choices: Sequence[Choice], data: SelectionInput, params: OP
    ) -> Sequence[int]: ...
```

(`SP` and `OP` are type variables bound to `SelectorParams` and
`OrdererParams`.)

- **`select`** returns the chosen fields of every well, in the order it
  ranks them (the order in which `counts` are attributed, §5.6). It may
  choose any centre (the scan's frames, route's lattice, anything else);
  the layer checks them (§5.5).
- **`order`** returns a permutation of `range(len(choices))`: the visiting
  order of the whole plate, across wells. It may reorder the wells.
- **Pure**: deterministic for the same input (a randomised method takes a
  `seed` in its params), no I/O but `data.scan.image(frame)`, no clock, no
  import of the hardware layer. NumPy is there; another library is a
  declared dependency in the same PR (SciPy: `scipy>=1.11.4`, the floor
  scikit-image already sets).
- A selector that cannot use its input raises `SelectionError` with a
  sentence the owner can act on.

### 5.4 The registry and the built-ins

`registry.py`:

```python
SELECTOR_GROUP = "smc.selectors"
ORDERER_GROUP = "smc.orderers"


def selectors() -> dict[str, Selector[Any]]: ...
def orderers() -> dict[str, Orderer[Any]]: ...
def selector(name: str) -> Selector[Any]: ...
def orderer(name: str) -> Orderer[Any]: ...
```

- An entry point's value names a class instantiated with no argument; its
  `name` equals the entry point's name and is kebab-case (m3's
  `NAME_PATTERN`); `summary` is one line; `Params` subclasses
  `SelectorParams` or `OrdererParams`. An entry point that fails to load or
  breaks these rules is left out with a logged warning, and `selector(name)`
  for it raises `SelectionError` giving the load error.
- An unknown name raises `SelectionError` that lists the known names,
  sorted.
- Registered in `pyproject.toml`:

```toml
[project.entry-points."smc.selectors"]
most-cells = "smc.selection.builtin:MostCells"

[project.entry-points."smc.orderers"]
nearest = "smc.selection.builtin:Nearest"
as-chosen = "smc.selection.builtin:AsChosen"
```

- **`most-cells`** is m3 §11's rule unchanged: per well in the plan's
  order, `choose_fields` on route's lattice; params `min_cells: int = 1`
  (≥ 1) and `candidate_step_fraction: float = 0.25` (> 0, ≤ 1). Its choices
  carry `score=None` and no measures, so the default route's JSON is
  unchanged.
- **`nearest`** is m3 §11's order unchanged: the wells in the plan's order;
  in each, nearest neighbour then 2-opt (`order_route`), starting at the
  choice nearest the previous well's last stop, or the well's
  `first_field` for the first well. No params.
- **`as-chosen`** keeps the selector's order (the identity). No params.

### 5.5 What the layer checks

`checks.py`, called by `route` on every result, raising `SelectionError`
that names the selector or orderer and the offending choice:

```python
def check_choices(choices: Sequence[Choice], data: SelectionInput) -> None: ...
def check_order(order: Sequence[int], n: int) -> None: ...
```

- each choice's well is one of `data.scan.wells`; `x_um`, `y_um`, `score`
  and every measure are finite; measure keys match `[a-z][a-z0-9_]*`;
- each choice is `data.allowed` (inside the well shrunk by the wall margin,
  away from the excluded centre);
- no well has more than `data.fields_per_well` choices;
- no two choices of one well overlap (route's rule: `|d_col| < W − 1e-6`
  and `|d_row| < H − 1e-6` in the imaging field's pixels; touching is
  allowed);
- the order is a permutation of `range(n)`.

A selector's mistake is a refusal, never a silent drop: the stage ends
`failed` with the check's sentence, and the collaborator sees it.

### 5.6 `route` with a selector and an orderer (amends m3 §11)

`RouteParams` gains four fields; the others are unchanged:

```python
class RouteParams(PluginParams):
    selector: str = "most-cells"
    selector_params: dict[str, JsonValue] = Field(default_factory=dict)
    orderer: str = "nearest"
    orderer_params: dict[str, JsonValue] = Field(default_factory=dict)
```

(`labels`, `fields_per_well`, `min_cells`, `exclude_centre_um`,
`wall_margin_um`, `objective` and `candidate_step_fraction` stay as m3 §11
has them.)

- **Validation** (an `after` model validator, so `--param` reports it as a
  `ParamsError`): the names are registered; `selector_params` validates
  against the selector's `Params` and `orderer_params` against the
  orderer's. `min_cells` and `candidate_step_fraction` are `most-cells`'
  parameters, kept on `RouteParams` for the presets and the page: with
  `most-cells`, they are its params, and giving either one in
  `selector_params` too is refused; with another selector, a value other
  than their default is refused, naming `most-cells`.
- `--param selector_params='{"min_density_per_mm2": 20}'` takes JSON (m3
  §1.6, a non-scalar field).
- **The run**: after the upstreams and the imaging field (unchanged), route
  builds `Scan.from_results(...)` and a `SelectionInput`; calls
  `select`; `check_choices`; attributes `counts` in the selector's order
  (each wanted cell to the first choice that holds it, as today); calls
  `order`; `check_order`; then numbers, names, focuses and measures the
  stops as m3 §11 and §26 do (`order`, `<well>_r<k:03d>`, `stop_focus`,
  `uncovered`, `distance_um`, `travel_s`). No stop at all is the
  `PluginRunError` it is today.
- A `SelectionError` from `select`, `order` or a check, and any other
  exception from a selector or an orderer, ends the stage as a
  `PluginRunError` whose message starts with `selector <name>:` or `orderer
  <name>:` (another exception's type is named too); the stage is `failed`.
- `RoutePosition` (m3 §3.5) gains `score: float | None = None` and
  `measures: dict[str, float] = {}`, both omitted from the JSON when `None`
  or empty; `nis-export` ignores them. The default route's result is
  byte-identical to today's.

### 5.7 Tests of a selector

`smc.selection.testing.make_input(...)` builds a `SelectionInput` in memory,
with no file and no stand, from cell positions:

```python
def make_input(
    cells: Sequence[tuple[str, float, float, str]],
    *,
    wells: Sequence[str] = ("B2",),
    plate: str = "96-well",
    field_um: tuple[float, float] = (100.0, 100.0),
    labels: Sequence[str] = ("single",),
    fields_per_well: int = 10,
    wall_margin_um: float = 0.0,
    exclude_centre_um: float = 0.0,
) -> SelectionInput: ...
```

Each cell is `(well, x_um, y_um, label)`, in stage µm; the plate's wells
come from useq's plate (`plate`, A1 at the origin); the field is a camera
with no rotation and no mirror (`field_um`, `(width, height)`, over a
100 × 100 px frame). `labels` is the input's `labels` (route's default,
`("single",)`, when not given: planning #567 added it, since a selector that
counts doublets is tested with them). It is in `smc.selection`, not
`smc.testing`, so it never imports `pymmcore_plus`. `image()` on such an
input raises `DatasetError` (no frames); a selector that reads pixels is
tested on a run folder written by the test.

### 5.8 Refined while planning #567 (2026-11-18)

Measured on the main checkout's `.venv` (Python 3.13.0, pydantic 2.13.5,
numpy as pinned) with throwaway prototypes, nothing committed; the steps and
the test names are in the plan of issue #567.

- **How route gets the document.** `Scan.from_results` takes a `RunDocument`
  (§3) and `RunContext` holds one only privately: the snapshot the runner took
  before `run`, which `upstream` reads. Decided: `RunContext.document`, a
  read-only property that returns that snapshot (additive to m3 §1.4, as
  `run_doc` was for #195). Route hands it to `Scan.from_results`, so a
  selector sees the same `scan.document` online and on a dataset. It adds no
  read that the runner has not checked: a selector that reads another stage's
  record through `scan.document` makes a read that staleness does not see, and
  the guide (§7.6) says to use the fields of `SelectionInput` and `Scan` only.
- **One definition of "inside".** `choose_fields` decides that a cell is in a
  field from the inverse that `_inverse_of` reads from `pixel_of_stage` at one
  µm along each axis, and `|d_col| <= W/2`, `|d_row| <= H/2` (m3 §11, FM-132).
  Testing `pixel_of_stage` per cell against the same half-extents disagreed with
  it on 51 of 800 cells placed on a field's edge. So everything that counts for
  a selector's choices uses `_inverse_of` and `_pixel_offsets` and never
  `pixel_of_stage` per cell. `route.algorithm` gains three public functions on
  them, and `choose_fields` is not changed:
  `cells_in_field(cells, x_um, y_um, *, matrix, image_shape_px)` (§5.2);
  `held_counts(cells, centres, *, labels, matrix, image_shape_px)`, one
  `dict[str, int]` per centre, in order, of the cells of `labels` that no
  earlier centre holds, zeros included (route's `counts`: on 120 wells,
  rotated and mirrored, three labels, it gave `choose_fields`' counts with no
  difference); and `first_overlap(centres, *, matrix, image_shape_px)`, the
  first pair `(i, j)`, `i < j`, whose fields overlap by §5.5's rule, or
  `None`, in numpy (a pair loop in Python is 125 000 pairs at 500 fields).
- **What route does, in this order** (the first progress and the first stop
  check of m3 §11 stay where tests pin them, per well): the upstreams, the
  labels and the imaging field as today; the cells file; `ctx.document`; one
  `select` for every well; `check_choices`; `held_counts` per well, in the
  selector's order; one `order`; `check_order`; then, for each well of the
  plan's order, `ctx.raise_if_stopped()`, the stops of that well in visiting
  order with their focus, and `ctx.progress(done, total, well)`; last, the
  positions sorted by their place in `order`, numbered `0..n-1`, named
  `<well>_r<k:03d>` with `k` the stop's rank among its well's stops in
  visiting order. `select` is one step that a stop cannot interrupt; at the
  measured 0.3 to 0.45 s a well that is 30 to 45 s for a 96-well plate where
  every well is dense.
- **`most-cells` is route's rule to the letter.** It skips a well with no
  wanted cell (so a lattice too large for such a well is no refusal, as now),
  takes the well's geometry from `scan.geometry`, derives the step from
  `candidate_step_fraction * min(fov_um)` of the imaging field, and turns
  `LatticeTooLarge`, `TooMuchWork` and `ValueError` into `SelectionError` with
  the plugin's present texts, so the stage's message is `selector most-cells:`
  followed by them.
- **`nearest` is route's order to the letter.** Wells in the plan's order; the
  first well that has a choice starts at the choice nearest its own
  `first_field`, every later one at the choice nearest the previous well's
  last stop (a well with no choice sets no reference); ties go to the choice
  that comes first in `choices`.
- **What `assumed` says.** The default route's `assumed` is unchanged. With a
  selector other than `most-cells`, one line is added after the existing
  ones: `the fields were chosen by selector '<name>', not by route's own rule
  of most cells: its choice, its score and its measures are its own`. With an
  orderer other than `nearest`, the line that says the route minimises the
  Euclidean distance is replaced by `the visiting order is orderer '<name>''s,
  not route's shortest path: distance_um is the length of that order's path,
  and whether the stage's move time follows it is not measured`.
- **The registry.** A name is looked up without importing anything (the
  entry points' names), so a bad name is refused with the known ones listed
  and the default `RouteParams()` imports only `most-cells`. `selector(name)`
  loads only the entries of that name; `selectors()` loads all, and a failure
  is a logged warning there and a `SelectionError` carrying the reason in
  `selector(name)` (FM-113). Entry points are metadata written at install
  time (FM-112): when no selector at all is registered, the message says to
  run `pip install -e .` in the smc folder. Tests put their own selectors in
  place of the installed ones by replacing the module's `entry_points`, as
  `tests/test_cli_presets.py` does for plugins.
- **`RouteParams`.** The names are checked by field validators (so a bad name
  and a bad bound are listed together), the params by an `after` model
  validator, which turns every exception of a selector's `Params` into a
  `ValueError` that starts with `selector_params:` or `orderer_params:`
  (FM-128). `selector_params` and `orderer_params` are `dict[str, JsonValue]`:
  pydantic's `JsonValue` refuses `NaN` and `Infinity` under `allow_inf_nan=False`
  (measured). A frozen model with a dict field is not hashable (measured), and
  nothing in `src/` hashes a plugin's params.

## 6. The drawing (#568)

`smc dataset draw RUN_DOC [--out DIR] [--wells A1,B2] [--max-px 1600]`,
the drawing in `src/smc/data/draw.py`, moved from
`scripts/dev/select_fields.py` (`canvas_of`, `place_frame`,
`stretch_to_uint8`, `render_png`, and what they use), which then imports
it.

- One PNG per well, `<DIR>/<well>.png`, default `DIR` = `<run
  folder>/draw/`: the frames' pixels block-meaned onto the well's canvas
  (longest side at most `--max-px`), the well's wall, the cells coloured by
  label, the `route` slot's stops of that well as footprints (the route's
  own matrix and shape, FM-124) numbered `order + 1`, the path between
  them, and the score after the number when the stop has one; a caption
  with the well, the route record's plugin, `selector` and `orderer`, the
  number of stops and the uncovered cells.
- Without a usable `route` record (none, or not usable on the scan's
  profile), the frames and the cells only, and the caption says why.
- The PNGs it writes are overwritten; nothing else in `DIR` is touched.
  Exit 0; 1 a PNG not written; 2 when `load_scan` refuses or `--wells`
  names a well the scan lacks.
- `pillow>=10.1` becomes a declared dependency (`ImageFont.load_default(size=…)`
  needs 10.1).

## 7. Two people in one repository (#554)

### 7.1 Where things go

An `area: algorithms` change lives in `src/smc/selection/` (its module),
`tests/` (its tests), `docs/algorithms/` (its page) and `pyproject.toml`
(its entry point; a dependency if it needs one). It does not touch
`src/smc/hardware/`, `profiles/` or `src/smc/ui/`; `/review` checks the
paths.

### 7.2 Two issues per result

- **The algorithm issue** (`area: algorithms`, the collaborator's, from the
  *Algorithm* template, §7.3) is done when the algorithm passes its tests
  in CI and has been looked at on a dataset with the owner.
- **The wiring issue** (`area: plugins` or `area: ui`, the owner's) names
  the first on its plan's `**Depends on**` line, connects the result to the
  tool that drives the microscope, and carries the check at a stand.

### 7.3 The issue template

`.github/ISSUE_TEMPLATE/algorithm.yml`, labels `type: feature` and `area:
algorithms`, fields: **Goal**; **Data in** (the dataset on the share, the
stages it needs); **Decision out** (the selector or orderer, its params,
what its measures hold); **How it is judged** (tests on synthetic inputs,
the dataset's drawings looked at with the owner, the numbers to report);
**The wiring issue** (the owner's, filled when it exists).
`scripts/github/labels.json` gains `area: algorithms` (colour `5319E7`,
"Pure algorithm on acquired data — no microscope, no smc.hardware (ADR-0004's
algorithm half)"), as it already is on GitHub.

### 7.4 Reviews and `main`

`.github/CODEOWNERS`:

```text
/src/smc/hardware/  @clementhelsens
/profiles/          @clementhelsens
/docs/adr/          @clementhelsens
/docs/design/       @clementhelsens
/.github/           @clementhelsens
/.claude/           @clementhelsens
/CLAUDE.md          @clementhelsens
```

Branch protection of `main`, set by the owner in the repository's settings
(not by a PR): a pull request is required; the light set and *closing
references* are required checks; **review from code owners is required**,
with no general approval count, so a PR on the paths above needs the
owner's approval and any other needs CI only. The owner remains the only
one who merges (CLAUDE.md applies to people too); the owner, as an
administrator, may bypass the rule for their own PRs on those paths, since
an author cannot approve their own PR.

### 7.5 Sessions

The collaborator has write access and runs this repository's skills in
Claude Code: `/develop <issue>` on an `area: algorithms` issue labelled
`status: ready`, `/feature` for a small one, `/adversarial-review` inside
`/develop` as for anyone. `/point`, `/lane`, `/plan`, `/review`, `/adr` and
`/hardware-session` stay the owner's. Their comments carry the same headers
(`## Report (develop session, <date>)`), which `point.py` reads as an
agent's. Their own notes start with no header, as the owner's do; `/point`
tells them apart by author.

### 7.6 Set-up on a laptop, and the guide

`CONTRIBUTING.md` gains a section *Working on algorithms (no microscope)*:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # pymmcore-plus comes as a library; no adapters
pre-commit install
pytest                           # the demo tests skip: "Micro-Manager demo adapters not installed"
pip install -e ".[dev,detect]"   # only to run the cell model (torch)
```

No `mmcore install`, no `smc doctor`, no vendor software. The demo tests
skip locally and run in CI (`SMC_REQUIRE_MM=1`); never set
`SMC_REQUIRE_MM` on the laptop. Building the acquisition loop (#556) needs
the demo adapters: `mmcore install --test-adapters`.

`docs/algorithms/README.md`, the guide, in this order: what the kit is for
(§0); set-up (above); get a dataset (the share's folder, unzip under
`local/`, `smc dataset check`, `smc dataset show`); load it in Python
(`load_scan`, frames, images, cells, `cells_table`); run the chain
offline (`smc run cells --offline` with the model or `detector=blobs`,
`density`, `route` with `selector=…`, `smc dataset draw`, `report`,
`nis-export`); write a selector (`/new-selector <name>`, or by hand: the
module, the entry point, the test, the page); test it (`make_input` cases
whose answer is known, then the dataset); what never to assume (a stage
position is a readback; a pixel size of 0 is unknown; a frame's `focus`
may be `unlocked`; the field chosen at another objective has that
objective's footprint); what never to touch (§7.1); the issue pair and
the PR (§7.2); what comes next (#556, #32). A page per algorithm,
`docs/algorithms/<name>.md`: what it measures, its params, what was tried
on which dataset, with numbers.

The skill `.claude/skills/new-selector/SKILL.md` (`/new-selector <name>`)
scaffolds `src/smc/selection/<module>.py` (its `Params`, a `Selector` whose
`select` returns the allowed scan frames' centres with the most wanted
cells, a working example to replace), `tests/test_selector_<module>.py`
(two `make_input` cases), the entry-point line, and
`docs/algorithms/<name>.md`.

## 8. Tests that hold the kit together

- `tests/test_kit_imports.py::test_kit_never_imports_pymmcore_plus` — a
  subprocess imports `smc.data` and `smc.selection` (and, from #568,
  `smc.data.draw`) and finds no `pymmcore_plus` in `sys.modules`.
- `tests/test_kit_imports.py::test_kit_modules_never_import_the_hardware_layer`
  — an AST walk over `src/smc/data/` and `src/smc/selection/` finds no
  import of `smc.hardware`, `smc.testing`, `smc.ui`, `pymmcore_plus` or
  `pymmcore`. #564 creates the module with its package; #567 adds its
  package to both lists.
- Each issue's own tests are named in its plan.

## 9. The issues and their order

| Issue | What | Depends on | Edits |
|---|---|---|---|
| #564 | §3 loader, `smc dataset show`, the import guard | — | `src/smc/data/`, `src/smc/cli.py` (one line) |
| #566 | §4 offline runs | — | `src/smc/cli.py` (`run`), `src/smc/hardware/microscope.py`, `src/smc/run/document.py`, the five plugins' classes, `route/plugin.py` |
| #565 | §2 export and check | #564 | `src/smc/data/` |
| #567 | §5 selection contract | #564, #566 | `src/smc/selection/`, `route/`, `slots/route.py`, `pyproject.toml` |
| #568 | §6 drawing | #565, #567 | `src/smc/data/`, `scripts/dev/select_fields.py`, `pyproject.toml` |
| #554 | §7 guide, template, CODEOWNERS, labels, skill | #564–#568 | `docs/`, `.github/`, `.claude/skills/`, `CONTRIBUTING.md`, `CLAUDE.md` |
| #570 | the first task (collaborator) | #554, a dataset of 2026-11-19 | `src/smc/selection/<name>.py` |
| #569 | the page's choice (owner), m5 §21 | #567 | `src/smc/ui/`, `src/smc/plugins/report/render.py`, `src/smc/plugins/slots/route.py` |

#564 and #566 both edit `src/smc/cli.py` and do not run at the same time.

## 10. Not decided here

- Whether a selector may read the 40× frames of another run to judge its
  choice (the first task may want it; a second `Scan` in the input is the
  likely shape, decided when #570 needs it).
- A decision during an acquisition (#556, #32): the contract here is for
  a choice made after a scan.
- pandas as a dependency: not now; `cells_table` gives the columns.
