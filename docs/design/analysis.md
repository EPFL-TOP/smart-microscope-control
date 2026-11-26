# Analysis design — `src/smc/analysis/`, the analysis track's package, and its first module: the review of fields

- **Status**: design for issue #654 (2026-11-25, decided by the owner, to be
  agreed by the collaborator in its pull request, ADR-0010), built by
  #676 (§1, §2.1 to §2.6) and #677 (§2.7 to §2.9).
- **Owner**: the design session; the document is the owner's to decide, like
  every file of `docs/design/` (ADR-0010). **Executors**: the analysis
  track's `/develop` sessions, planned by its lead.
- **Rule**: as in the other design documents, this one is the contract
  between issues built in parallel. Names, module paths and signatures below
  are fixed; an executor who needs to change one stops and comments on its
  issue (`## Plan deviation`). M1's conventions and CLI rules hold where this
  document does not say otherwise.

Read first: [ADR-0010](../adr/0010-two-tracks.md) (two tracks; the
conditions on an interface of the analysis track) and
[`algorithms.md`](algorithms.md) §2 (the dataset), §3 (the loader) and §7
(where things go).

## 0. Scope

`src/smc/analysis/` holds the analysis track's own modules: what works on
acquired data and is not a selection algorithm (those stay in
`src/smc/selection/`, `algorithms.md` §5). Its interface is in it too: the
pages a person uses to look at a scan, judge it and compare what tools
found, served beside the plate page and never inside it.

The first module is **the review of fields** (§2): a person goes through the
fields of a scan, sees each frame with the cells found on it, and says which
fields they want. The later modules are named in §3 and designed when their
issues come.

Out of scope: anything that drives or needs a microscope (the stand track's),
the plate page and its screens (`src/smc/ui/`, m5), and the question of a
notebook as a front of smc (#632, §4).

## 1. The package

### 1.1 Rules

- **It reads acquired data through the interfaces between the tracks**:
  `smc.data` (`load_scan`, `Scan`), the models of `smc.plugins.slots`
  (`ScanFrame`, `Cell`, `CellsResult`) and `smc.run` (`RunDocument`,
  `smc.run.files.write_atomically`). It imports nothing of `smc.hardware`,
  `smc.testing`, `pymmcore_plus` or `pymmcore`, and of `smc.ui` only the view
  components that the stand track makes free of the hardware layer for it
  (§2.8); never `smc.ui.controller` or `smc.ui.plate_page`.
- **It never writes a stage record** (`smc.run.write_stage`). The run
  document is the stand's record of what the tools did, and a dataset's
  `smc dataset check` would report its `run.json` as changed. A module writes
  only under a folder of the run folder that it names (`review/` for §2).
- **A module's pure part imports no Panel**: its models, files and state are
  tested everywhere and used from a script or a notebook. The Panel part is
  one module per view, `<module>_view.py`, imported by its command inside
  the command.
- **Panel comes from the `ui` extra** (`panel>=1.9,<2`, the page's pin): no
  extra of its own, since a second pin of the same library could disagree
  with the first. The pure part needs only the base install.
- **Its commands are a `typer` sub-app, `smc analysis`**, in
  `src/smc/analysis/cli.py`, added to `src/smc/cli.py` by one line,
  `app.add_typer(analysis_app, name="analysis")`. That line is the one
  crossing of this design onto a stand path: it is agreed here, and the pull
  request that adds it asks for the owner's approval (#677).
  `smc.analysis.cli` loads neither Panel nor Pillow when it is imported, so
  `smc --help` and `import smc.cli` stay as light as today
  (`tests/test_kit_imports.py` already checks that `smc.cli` loads no PIL).

`src/smc/analysis/__init__.py` re-exports the review's public names
(`ReviewSession`, `FieldsReview`, `FieldJudgement`, `Verdict`,
`FieldFilter`, `ReviewField`, `WellCount`, `ReviewError`), never a view.

### 1.2 The import guard

`tests/analysis/test_analysis_imports.py`, written by #676:

- `test_analysis_modules_never_import_the_hardware_layer`: an AST walk over
  every `.py` of `src/smc/analysis/`, refusing relative imports as
  `tests/test_kit_imports.py` does, finds no import of `smc.hardware`,
  `smc.testing`, `smc.ui`, `pymmcore_plus` or `pymmcore` (a name or any of
  its submodules). The test holds a tuple `UI_VIEWS_ALLOWED`, empty in
  #676; #677 puts in it exactly the module that #631 names
  for its image view (§2.8), and nothing else of `smc.ui`.
- `test_the_review_loads_no_panel_and_no_pymmcore_plus`: a subprocess imports
  `smc.analysis` and `smc.analysis.review` and finds neither `panel` nor
  `pymmcore_plus` in `sys.modules`. (`smc.hardware` itself may be there:
  `smc.plugins.slots` imports `smc.hardware.microscope`, `algorithms.md` §1.)

`tests/analysis/` has no `__init__.py`, as `tests/` has none: every module's
basename is unique under `tests/`. A module there imports the helpers of
`tests/` (`select_fields_seed.seed_run`) as `tests/unit/` does, through
`tests/conftest.py`'s folder on `sys.path` (measured, §2.2).

## 2. The review of fields

### 2.1 The task

A person (either lead, a biologist) opens a scan with no microscope: a run
folder at the stand, or a dataset on a laptop. They go through its fields
well by well. For each field they see its frame at working size, with the
cells that the `cells` stage found drawn as boxes in their class's colour,
with the class and the score; and they say whether it is a field they want:
**wanted**, **not wanted** or **unsure**, with a note if they wish.

What "wanted" means is the review's **purpose**, a sentence written when the
review starts ("fields with at least three single cells and no debris, for
the 40×") and kept in the file, so that the verdicts can be read later by
someone else.

What it is for:

- **A ground truth for a selection.** A selector (#570; #636's
  `low-mag-survey`) is judged by the fields it chose against the fields a
  person wanted. That comparison is a later module (§3); the review makes the
  wanted fields exist.
- **Looking at what a model found** on every field of a scan, which #600 asks
  for ("quand on passe en revue les FOV"). Choosing a model by name comes with
  #600 (§3); the review records which detector's cells were shown.

The review does not correct the cells (a later module, §3) and does not
choose fields to image (`route`'s job).

### 2.2 Measured (2026-11-25, planning #654)

On `main` at `1559560`, the main checkout's `.venv` (Python 3.13, pydantic
2.13.5, Panel 1.10.0), with a scan seeded by `tests/select_fields_seed.py`'s
`seed_run` (wells A1 and B2, 2 × 2 fields each, one cell):

- **A field has two numbers.** `ScanFrame.index` is the field's number
  within its well; `Cell.frame` and `Scan.cells_of(frame=…)` use the
  **position**, the index into `ScanResult.frames`. B2's first field has
  `index` 0 and position 4 (FM-248). The review shows a field by its
  position and keys a judgement by the frame's `file`.
- `ScanFrame.file` is the frame's path relative to the run folder, in POSIX
  form (`scan/t/A1_0000.tif`); a dataset keeps the same relative paths
  (`algorithms.md` §2).
- The `scan` stage record's `started_utc` is an `AwareDatetime`.
  `scrub_document` rewrites only `params`, `result`, `checkpoint` and `error`,
  so a dataset and the run folder it came from carry the same time. A pydantic
  model with an `AwareDatetime` field comes back equal from
  `model_dump_json` and `model_validate_json`, and a naive time is refused.
- `smc dataset check` ignores the files of the folder that its manifest does
  not list ("the collaborator's own outputs live beside the dataset"), so a
  `review/` folder in a dataset leaves the check green. `export_dataset` copies
  the frames and the cells file only: a review does not travel with an export.
- The detector is named by the `cells` result only: `CellsResult.model` (the
  weights' file name, or `blobs`) and `model_sha256`; `Cell` has no model
  field. The seeded scan's model is `m.pth`, its classes `single`, `doublet`,
  `debris`.
- **No module draws one frame's pixels with its cells' boxes.**
  `smc.data.draw` draws a well's mosaic with the cells as dots; the page's
  `image_figure` draws a frame with no box. `smc/ui/__init__.py` imports
  nothing; `smc.ui.controller` imports `smc.hardware` at module level.
- `smc.run.files.write_atomically(path, data, *, make_parents=True)` writes
  a temporary file in the same folder and `os.replace`s it onto the target
  (FM-38, FM-115).
- Nothing in smc writes a person's judgement of a field or of a cell today.
  Cell corrections are made in another repository's page, on the COCO export
  of `scripts/dev/cells_check.py`.
- A test module under a new `tests/analysis/` imports
  `select_fields_seed` and passes (a probe, removed).

### 2.3 What it reads

- `load_scan(run_doc)` gives the `Scan`; a `DatasetError` ends the opening
  (the command exits 2).
- **The fields** are `scan.result.frames`, in the scan's order, grouped by
  `scan.wells` (the plan's order). A well with no frame is listed with no
  field. A failed field (`ScanResult.failed`) has no frame and is not listed.
- **The cells of a field** are `scan.cells_of(frame=position)` when
  `scan.cells` is not `None`. Otherwise the field has no box, and the view
  says `scan.cells_absent`.
- **The classes** are `scan.cells_result.classes`; **the detector** is
  `scan.cells_result.model` and `.model_sha256`.
- **A frame's pixels** are `scan.image(frame)`, read when the field is shown,
  never at open. A `DatasetError` there is shown on that field; the review
  goes on, and the field can still be judged (a broken frame is a field one
  does not want).
- **The scan's identity** is `scan.document.provider("scan").started_utc`.

### 2.4 What it writes: the review file

`<run folder>/review/fields-<reviewer>.json`, **one file per reviewer**, so
that two people review the same scan without writing the same file. The
reviewer is a slug matching `^[a-z0-9][a-z0-9-]{0,31}$`, always given: a
name that travels with a dataset is chosen, never read from the PC's login.

```python
REVIEW_FOLDER = "review"


class Verdict(str, Enum):
    """A person's judgement of one field; Python 3.10 has no StrEnum."""

    WANTED = "wanted"
    NOT_WANTED = "not-wanted"
    UNSURE = "unsure"


class FieldJudgement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    well: str
    name: str
    verdict: Verdict
    note: str = ""
    cells_model: str | None
    cells_model_sha256: str | None
    judged_utc: AwareDatetime


class FieldsReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    reviewer: str
    purpose: str
    scan_started_utc: AwareDatetime
    judgements: tuple[FieldJudgement, ...] = ()


class ReviewError(ValueError):
    """A review that cannot be opened or saved; the message names the file and why."""


def review_path(run_dir: Path, reviewer: str) -> Path: ...
def load_review(path: Path) -> FieldsReview: ...
def save_review(path: Path, review: FieldsReview) -> None: ...
```

- `file`, `well` and `name` are the frame's own (`ScanFrame`); `file` is the
  key: **one judgement per file**. A new verdict on a judged field replaces
  its judgement in place; a first verdict is appended. The file keeps no
  history.
- `cells_model` and `cells_model_sha256` name the detector whose cells were
  shown when the verdict was given, `None` and `None` when none were. A
  verdict given on another model's boxes can be told apart once the cells
  are run again, or once #600 shows several models.
- `judged_utc` and `scan_started_utc` are aware UTC times.
- `review_path` raises `ReviewError` for a reviewer that is not a slug
  (`""`, `"Ana B"`, `"../x"`).
- `load_review` raises `ReviewError` naming the file for a file that cannot
  be read, is not JSON, or does not validate (with pydantic's first error).
- `save_review` writes `review.model_dump_json(indent=2) + "\n"`, encoded in
  UTF-8, with `write_atomically` (imported into `smc.analysis.review`'s
  namespace, so that a test replaces it there); its `OSError` becomes a
  `ReviewError` naming the file and the error.

The run document is never written. The review file is the review's only
state on disk.

### 2.5 The state: `ReviewSession`

`src/smc/analysis/review.py`, with the models above:

```python
@dataclass(frozen=True, slots=True)
class FieldFilter:
    well: str | None = None
    verdict: Verdict | Literal["unjudged"] | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewField:
    position: int
    frame: ScanFrame
    cells: tuple[Cell, ...]
    judgement: FieldJudgement | None


@dataclass(frozen=True, slots=True)
class WellCount:
    well: str
    fields: int
    wanted: int
    not_wanted: int
    unsure: int

    @property
    def unjudged(self) -> int: ...


class ReviewSession:
    scan: Scan
    path: Path

    @classmethod
    def open(
        cls,
        run_doc: Path | str,
        *,
        reviewer: str,
        purpose: str | None = None,
        clock: Callable[[], datetime] = ...,
    ) -> ReviewSession: ...

    @property
    def review(self) -> FieldsReview: ...
    @property
    def unknown_files(self) -> tuple[str, ...]: ...
    def fields(self, flt: FieldFilter | None = None) -> tuple[ReviewField, ...]: ...
    def field(self, position: int) -> ReviewField: ...
    def next_position(
        self, position: int | None, flt: FieldFilter | None = None
    ) -> int | None: ...
    def counts(self) -> tuple[WellCount, ...]: ...
    def judge(
        self, position: int, verdict: Verdict, note: str = ""
    ) -> FieldJudgement: ...
    def clear(self, position: int) -> None: ...
```

**Opening** (`open`):

1. `load_scan(run_doc)`; its `DatasetError` propagates.
2. The path is `review_path(scan.run_dir, reviewer)`.
3. **An existing file** is loaded. Its `scan_started_utc` must equal the scan
   record's `started_utc`, compared as times, never as strings; otherwise
   `ReviewError` naming the file and both times: its verdicts were given on
   another scan's frames, since a scan run again writes new files under a
   new token (FM-244). A `purpose` given and different from the file's is a
   `ReviewError` ("this review's purpose is …; choose another reviewer name
   for another purpose"); none given takes the file's.
4. **A missing file** needs a non-empty `purpose` (`ReviewError` otherwise)
   and is written at once, with no judgement, so that a folder that cannot
   be written is found when the review opens and not at its first verdict.
5. The session keeps the SHA-256 of the bytes it read or wrote.

**Reading**:

- `fields(flt)` lists the fields in the scan's order (`None` or an empty
  filter: every field); `well` keeps the
  fields of that well (`ReviewError` for a well the scan does not have),
  `verdict` those with that verdict or, with `"unjudged"`, those with none,
  `label` those with at least one cell of that class (`ReviewError` for a
  class `cells_result.classes` lacks; none when the scan has no cells).
  `field(position)` raises `ReviewError` for a position outside the scan.
- `next_position(position, flt)` is the first field of `flt` after
  `position` in the scan's order (the first of all when `position` is
  `None`), or `None` when there is none: the view moves there after a
  verdict.
- `counts()` gives one `WellCount` per well of `scan.wells`, in that order,
  a well with no field included.
- `unknown_files` are the judgements whose `file` is no frame of the scan
  (only an edit by hand makes one, since the scan's time matched): kept in
  the file, never shown, counted on the view.

**Writing** (`judge` and `clear`), under one `threading.Lock` held from the
read to the swap, because every tab of the served page shares the session:

1. Read the file's bytes again. When they differ from the kept hash (another
   window, another program, a sync tool, the file deleted), write nothing and
   raise `ReviewError` ("… changed since this review read it; nothing
   written; open the review again").
2. Build the new `FieldsReview` (the judgement with `clock()` as
   `judged_utc`, the detector of §2.4; or without the field's judgement for
   `clear`, which is a no-op on an unjudged field and writes nothing).
3. `save_review`. A `ReviewError` there (a read-only share; Windows refusing
   to replace a file another program holds open, FM-24) propagates.
4. Only then swap the kept review and hash. A judgement that was not written
   is not kept in memory either: what the view shows is what the file holds.

A note is saved with a verdict, never alone. The `note` is stored as given,
stripped of surrounding white space.

### 2.6 Tests of the file and the state (#676)

`tests/analysis/test_analysis_review.py`, on scans seeded by `seed_run` in
`tmp_path` (wells A1 and B2, cells placed with `cell_at`):

- `test_a_new_review_needs_a_purpose_and_is_written_when_it_opens`
- `test_a_reviewer_name_that_is_not_a_slug_is_refused`
- `test_a_review_of_another_scan_is_refused_naming_both_times`
- `test_a_purpose_that_differs_from_the_files_is_refused`
- `test_a_judgement_is_on_disk_when_judge_returns` (a second session opened on
  the same file reads it)
- `test_a_second_verdict_replaces_the_first_and_clear_removes_it`
- `test_the_cells_of_a_field_are_those_of_its_position_not_of_its_index_in_the_well`
  (FM-248: a cell on B2's first field, position 4 and index 0, is on that
  field and on no field of A1)
- `test_the_judgement_names_the_detector_whose_cells_were_shown_and_none_without_cells`
- `test_a_scan_without_cells_is_reviewed_with_no_boxes_and_says_why`
- `test_a_file_changed_by_another_writer_is_not_overwritten_and_the_judgement_is_not_kept`
- `test_a_failed_write_raises_naming_the_file_and_keeps_the_last_saved_review`
  (`write_atomically` replaced in `smc.analysis.review` by one that raises
  `PermissionError`)
- `test_a_note_with_accents_comes_back_as_written` (`"cellules fusionnées"`,
  UTF-8 on every platform)
- `test_filters_by_well_by_verdict_by_unjudged_and_by_label`
- `test_next_position_follows_the_filter_and_ends_with_none`
- `test_counts_follow_the_plans_wells_and_include_a_well_with_no_field`
- `test_a_judgement_of_a_file_the_scan_lacks_is_kept_and_counted_never_shown`
- `test_a_review_folder_in_a_dataset_leaves_its_check_green` (`export_dataset`,
  a review written in the dataset, `check_dataset` still green: the review
  relies on §2.2's fact about `check`, and this test says so if it changes)

And §1.2's two tests in `tests/analysis/test_analysis_imports.py`.
`docs/algorithms/review.md` (new) says what the review is for, the file and
its fields, and how to read one in Python.

### 2.7 The view and the command (#677)

`src/smc/analysis/review_view.py`, the only module of the review that
imports Panel:

```python
class ReviewView:
    def __init__(self, session: ReviewSession) -> None: ...
    def panel(self) -> pn.viewable.Viewable: ...
```

At 1280 × 800, three columns:

- **Left** (about 240 px): the purpose; the wells, each with
  `judged / fields` and the number wanted (`counts()`); the filters (well,
  verdict with *unjudged*, class); how many fields the filter keeps; the
  unknown files' count when there are any.
- **Centre**: the frame, in #631's image view (§2.8), as large as the column
  allows, zoom and pan; the cells' boxes in their class's colour
  (`smc.data.draw.CLASS_COLOURS`, so that a drawing and the review agree),
  each with its class and its score to two decimals; a scale bar from
  `scan.pixel_size_um` (the scan's camera, FM-124). A frame that cannot be
  read shows the `DatasetError`'s text in its place.
- **Right** (about 280 px): the field (well, name, `n of N` in its well, its
  position, `focus`, `z_um`), its cells per class, the detector (`model` and
  the first 12 characters of its SHA-256); the buttons *Wanted*, *Not
  wanted*, *Unsure* and *Clear*; the note; *Previous* and *Next*.

Behaviour:

- A verdict saves (`judge`), then shows `next_position` of the filter; at the
  end of the filter the view stays and says that the filter has no field
  left.
- **A second verdict within 0.5 s of the one before is ignored** (FM-245): a
  double-click would otherwise judge the next field, which the reviewer has
  not seen.
- A `ReviewError` at a verdict shows on a red line under the buttons (*Not
  saved: …*) and the view stays on the field.
- Every tab of the server shares the one `ReviewSession`; each tab has its own
  field and filter.
- **In a notebook**, `pn.extension()` and then
  `ReviewView(ReviewSession.open(...)).panel()` show the same view in a cell;
  `docs/algorithms/review.md` has the example.

`smc analysis review RUN_DOC --reviewer NAME [--purpose TEXT] [--port 5007]`,
in `src/smc/analysis/cli.py`:

- opens the session; a `DatasetError` or a `ReviewError` prints its message
  and exits 2; without the `ui` extra it prints the advice `smc ui` prints and
  exits as `smc ui` does;
- serves the view with `pn.serve` on `localhost` only, port 5007 by default
  (the page's is 5006, so both run side by side), `show=False`; prints the
  URL and the review file's path; ends on Ctrl-C;
- has no `--profile` and no `--dry-run`: it opens no `Microscope` and reads no
  stand;
- imports `review_view` inside the command (`importlib.import_module`).

`src/smc/cli.py` gains the one line of §1.1.

### 2.8 What the view needs from #631 (asked of the stand track)

#631 builds the image view as a component of its own, which the page embeds
and a notebook shows. The review uses it as it is, and needs from it:

- a module of `smc.ui` that imports no `smc.hardware`, no `smc.ui.controller`
  and no `smc.ui.plate_page` (`smc/ui/__init__.py` imports nothing, §2.2), so
  that `smc.analysis` may import it;
- a 2-D frame as input (the camera's dtype with its bit depth, or `uint8`),
  the pixel size for its scale bar, and **boxes**: `(row0, col0, row1, col1)`
  in the frame's pixels, each with a colour and a text, drawn over the image
  and following its zoom and pan;
- the same component in a served page and in a notebook (#631's own
  acceptance criterion).

The module's name and the component's signature are #631's plan's to fix, in
m5; #677's plan copies them and adds that module to
`UI_VIEWS_ALLOWED` (§1.2). Until #631's component is merged, #677
waits; #676 does not.

### 2.9 Tests of the view and the command (#677)

- `tests/analysis/test_analysis_review_view.py`, with no browser: a verdict
  writes the file and shows the next field of the filter; a second verdict
  within 0.5 s is ignored; a `ReviewError` shows the red line and keeps the
  field; a filter changes the fields and the count; the boxes given to the
  image view are the cells of the field's position (FM-248).
- `tests/analysis/test_cli_analysis_review.py`: exit 2 for a run document
  `load_scan` refuses, for a reviewer that is not a slug, for a new review
  with no purpose; `smc analysis --help` lists `review`; a subprocess that
  imports `smc.cli` finds no `panel` in `sys.modules`.
- A look at the served page, at 1280 × 800: #677's plan names how
  (a Playwright screenshot as `/ui-check` takes them; `/ui-check` itself
  serves the plate page only).

### 2.10 Failure modes

- **FM-248** (two numberings): a field's position and its index in its well;
  §2.6's test.
- **FM-38**, **FM-115**: the file is written by `write_atomically`.
- **FM-24**: Windows refuses to replace a file another program holds open;
  the write fails as a `ReviewError` and nothing given before is lost.
- **FM-244**: what the review says is keyed by the scan record's time; a
  review of another scan is refused.
- **FM-124**: the scale bar is the scan's camera (`Scan` already holds it).
- **FM-245**: a double-click on a verdict.
- Text is read and written as UTF-8 (`PLW1514`); a note may hold accents.

## 3. Later modules, not designed here

- **Correcting the cells**: per box, the right class, another class or not a
  cell, and the cells a model missed; written in `scripts/dev/cells_check.py`'s
  COCO format, so that its `measure` reads it, and replacing the other
  repository's review page. Its own section and issue.
- **Several models' labels** (#600, after #599): the review chooses whose
  boxes it shows; `FieldJudgement.cells_model` already names them.
- **A selection against the verdicts**: the fields a selector or `route`
  chose against the wanted ones, per well; how a selector is judged.
- **A review carried back** to the stand's run folder: by hand (copy
  `review/`) until a need appears. An export that carries it is a change of
  `smc.data`, an issue of the stand track.
- Keyboard shortcuts for the verdicts.

## 4. Not decided here

- **#632** (a notebook as a front of smc): the review is a Panel component
  that also shows in a notebook, as #631's views do, and this decides nothing
  more.
- Whether a selector may read a review while it runs: not now; a review is
  read by the module that judges a selection (§3).

## 5. The issues and their order

| Issue | What | Depends on | Edits |
|---|---|---|---|
| #676 | §1, §2.3 to §2.6: the review file, the session, the import guard, the page in the guide | this design merged | `src/smc/analysis/__init__.py`, `review.py`; `tests/analysis/`; `docs/algorithms/review.md` |
| #677 | §2.7 to §2.9: the view and `smc analysis review` | #676, #631's image view | `src/smc/analysis/review_view.py`, `cli.py`; `src/smc/cli.py` (one line, the owner's approval); `tests/analysis/`; `docs/algorithms/review.md` |

Both are issues of the analysis track (`track: analysis`), planned by its
lead.
