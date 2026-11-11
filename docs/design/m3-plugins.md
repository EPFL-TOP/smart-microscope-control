# M3 design — plate tools as plugins

- **Status**: design for issue #18 (§1, 2026-10-26); the plate-scan
  workflow, §2 to §16, for issue #192 (2026-10-27); the plate from images of
  the walls, §17, for issue #328 (2026-11-05); objectives' offsets, §18,
  for issue #340 (2026-11-05); the sides of a wall, §19, for issue #327
  (2026-11-09); the focus on the cells, §21, for issue #341 (2026-11-09);
  the lab's own plate definitions, §22, for issue #400 (2026-11-10); a
  plan's calibration from another run document, §23, for issue #383
  (2026-11-10); the plate's orientation in its holder, §24, for issues #434
  and #422 (2026-11-11).
- **Owner**: the design session. **Executors**: `/develop` sessions, one per issue.
- **Rule**: as in M1 and M2, this document is the contract between issues
  that are built in parallel. Names, module paths and signatures below are
  fixed; an executor who needs to change one stops and comments on its
  issue instead. M1's conventions (§2), its CLI rules (§9) and its error
  taxonomy hold everywhere this document does not say otherwise.

Read first: [ADR-0004](../adr/0004-plugin-architecture.md) (tools are
plugins over capabilities), [ADR-0003](../adr/0003-capabilities-roles-and-profiles.md),
M1 §2, §7 (the facade) and §9 (the CLI).

## 0. Scope

M3 turns the plate tools of `nikon-control` into plugins: the plugin API
and the run document (#18), then one workflow on the Ti2-E, end to end
(re-scoped 2026-10-26, `docs/roadmap.md`): calibrate the camera and the
plate, choose wells, scan them at 40x, detect single and doublet cells,
count them per well, choose fields of view and the shortest route between
them, show it, and export it to NIS-Elements. §1 fixes #18. §2 to §16 fix
every interface of the workflow at once (#192), so that its issues are
planned and built in parallel; a plan may extend this document only
additively (a new optional `RunContext` method, a new optional field with
a schema bump, §1.7), and anything else comes back as a `## Plan
deviation`.

`nikon-control` had no run document in code, only the note
(`docs/acquisition-architecture.md`) and ad-hoc files: a `plate.json`
without a version, rewritten in place, and a corrupt one silently replaced
by `{}`; a `scan.json` rewritten per well. Nothing in it was atomic,
versioned, invalidated or resumable, so everything in §1.7 is new design,
not a port. Two of its facts shape §1: plate registration (#19) and plate
finding (#20) produce the same thing, which the wells read whichever
produced it (§1.3, *slots*); and timings (#24) and the camera calibration
(#22) are used by the scan when present, not required (§1.3,
`reads_optional`).

## 1. The plugin API (#18)

### 1.1 What was measured (2026-10-26, pydantic 2.13.5, typer 0.27.2, Python 3.13)

- A generic `Protocol[P, R]` with class attributes `Params: type[P]` and
  `Result: type[R]` and `run(self, microscope, params: P, ctx) -> R`
  passes `mypy --strict` with the repository's configuration when a
  concrete class sets `Params = StageReportParams` and types `run` with
  its own models, assigned to both `Plugin[StageReportParams,
  StageReportResult]` and `Plugin[Any, Any]`. ruff's `N` rules accept the
  `Params`/`Result` attribute names. `isinstance(obj, Plugin)` works at
  runtime with `@runtime_checkable` (data members included) and is
  `False` for `object()`.
- `ConfigDict(allow_inf_nan=False)` refuses `nan` and `inf` both from
  Python and from the strings `"nan"`, `"inf"`, `"NaN"` (error type
  `finite_number`).
- `Model.model_validate(raw)` in pydantic's default lax mode turns the
  strings of a command line into the field's type: `"3"` and `"3.0"` into
  `int` (`"3.5"` refused), `"1e3"` into `1000.0`, `"false"`, `"no"`,
  `"off"`, `"0"` into `False` (`"2"` refused), a `Path`, an `Enum` member
  and a `Literal` from their text; a `str` field keeps `"3"` as `"3"`. It
  does **not** turn a string into a list or tuple (`tuple_type`), and
  neither does `model_validate_strings`: a container field needs its value
  decoded first (§1.6). `extra="forbid"` refuses an unknown key
  (`extra_forbidden`).
- `model_dump(mode="json")` of a model holding a lone surrogate (an
  adapter's string, FM-05) succeeds, `json.dumps(..., ensure_ascii=True)`
  writes it as `\udc80`, and the round trip gives back an equal model;
  `model_dump_json()` of the same model raises
  `PydanticSerializationError`. A UTC `datetime` dumps as
  `2026-10-26T09:00:00Z`. A `useq.WellPlatePlan` field round-trips
  through `model_dump(mode="json")` and `model_validate`.
- `json.loads` of a file read with `encoding="utf-8"` fails on a
  byte-order mark (FM-80); `encoding="utf-8-sig"` reads it.
- `importlib.metadata.EntryPoint(name=..., value="module:Attr",
  group="smc.plugins").load()` returns the attribute of a module already
  in `sys.modules`; its `.dist` is `None`. The group is declared, empty,
  in `pyproject.toml`.
- `FakeCore.demo_like()` gives every capability; a profile with
  `[roles.exclude] xy_stage = ["XY"]` and `focus = ["Z"]` over it gives a
  facade without `XYStage` and `ZStage`, and `require(XYStage)` then
  raises a `CapabilityMissingError` whose text contains `[roles.assign]`
  (FM-36).

### 1.2 Module map

```
src/smc/plugins/
  __init__.py        re-exports the public names of api, errors, registry, params, runner
  api.py             PluginParams, PluginResult, Plugin, Progress, RunContext, StopRequested
  errors.py          PluginError and its subclasses (§1.9)
  registry.py        GROUP, PluginEntry, LoadFailure, Registry, discover(), load()
  params.py          parse_params(), describe_params()
  runner.py          check_ready(), run_plugin()
  stage_report.py    the built-in StageReport plugin (§1.11)
src/smc/run/
  __init__.py        re-exports document's public names
  document.py        SCHEMA_VERSION, StageStatus, ReadRecord, StageRecord,
                     RunDocument, RunDocumentError, write_stage()
src/smc/cli.py       `smc run` (§1.10)
```

`smc.run` imports nothing from `smc.plugins` (plugins depend on the run
document, not the reverse) and nothing from `smc.hardware`. Neither
package imports `pymmcore_plus` (TID251).

### 1.3 The protocol and its models — `smc/plugins/api.py`

```python
class PluginParams(BaseModel):
    """Base of every plugin's parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
```

```python
class PluginResult(BaseModel):
    """Base of every plugin's result."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    # What the result rests on that was assumed rather than measured.
    assumed: tuple[str, ...] = ()
```

```python
P = TypeVar("P", bound=PluginParams)
R = TypeVar("R", bound=PluginResult)


@runtime_checkable
class Plugin(Protocol[P, R]):
    name: str
    summary: str
    Params: type[P]
    Result: type[R]
    requires: tuple[type, ...]
    provides: str
    reads: tuple[str, ...]
    reads_optional: tuple[str, ...]

    def run(self, microscope: Microscope, params: P, ctx: RunContext) -> R: ...
```

- **`name`**: kebab-case, `NAME_PATTERN = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"`
  (full match), equal to the entry point's name. It keys the plugin's
  record in the run document and names its logger.
- **`summary`**: one line for `smc run --list`.
- **`Params`, `Result`**: subclasses of `PluginParams` and
  `PluginResult`. The bases make every float finite (FM-32: a bound the
  plugin forgot still refuses `nan`), refuse unknown keys, and freeze the
  model. Unknown is `None`, never `nan` ("unknown beats guessed"). A
  result field that says where to image is a `useq` object (ADR-0004).
- **`requires`**: capability Protocols, each a key of
  `smc.hardware.microscope.CAPABILITY_ROLES`.
- **Slots.** `provides` names the slot of the run document that the result
  fills, by the same pattern; by convention it is the plugin's name unless
  several plugins produce the same thing. #19 and #20 both provide
  `plate`, and #21 reads `plate` whichever plugin wrote it. The model of a
  slot belongs to the plugin that introduces it; another provider imports
  that `Result` (or a subclass of it) rather than redefining it.
- **`reads`**: slots the plugin needs; the runner refuses to start without
  them (§1.8). **`reads_optional`**: slots the plugin uses when they are
  usable (the scan's timings and camera calibration). The two are
  disjoint and neither holds `provides`.
- **`run`** is the hardware loop of ADR-0004; the pure algorithm lives in
  another module and takes no `Microscope`. It returns its `Result` and
  never writes the run document itself.

The entry point names a **class**, which the registry instantiates with no
argument: `stage-report = "smc.plugins.stage_report:StageReport"`. A
plugin with an algorithm is a package, `smc/plugins/<snake_name>/` with
`algorithm.py` and `plugin.py`, registered as
`<name> = "smc.plugins.<snake_name>.plugin:<ClassName>"`.

```python
@dataclass(frozen=True, slots=True)
class Progress:
    plugin: str
    done: int
    total: int | None
    message: str
```

```python
class StopRequested(Exception):
    """A plugin's way out when ``ctx.should_stop()`` is true."""
```

`StopRequested` derives from `Exception` and from nothing in
`smc.plugins.errors` or `smc.hardware.errors`, so an `except PluginError`
or `except SafetyRefusedError` in a loop never swallows a stop (FM-64).

### 1.4 `RunContext` — what a plugin gets besides the microscope

```python
class RunContext:
    plugin: str
    logger: logging.Logger
    run_dir: Path
    run_doc: Path
    dry_run: bool

    def progress(
        self, done: int, total: int | None = None, message: str = ""
    ) -> None: ...
    def should_stop(self) -> bool: ...
    def raise_if_stopped(self) -> None: ...
    def wait_s(self, seconds: float) -> None: ...
    def upstream(self, slot: str, model: type[M]) -> M: ...
    def upstream_or_none(self, slot: str, model: type[M]) -> M | None: ...
    def save_checkpoint(self, state: BaseModel) -> None: ...
    def checkpoint(self, model: type[M]) -> M | None: ...
```

`M = TypeVar("M", bound=BaseModel)`. The constructor is the runner's
(keyword-only, not public API): a plugin is tested through `run_plugin`
with a run document under `tmp_path`, never by building a context.

- **`logger`** is `logging.getLogger(f"smc.plugins.{name}")`. The context
  adds no handler and sets no level (FM-46); the CLI's `-v` shows it.
- **`run_dir`** is the folder of the run document. A plugin that writes
  files (the scan's frames, #21) writes them under it and stores their
  paths relative to it in its result, so the folder moves as a whole.
- **`run_doc`** is the path of the run document as the runner was given it
  (added for #195, which prints the file's name in its header; the context
  already held it, privately, for `save_checkpoint`). A plugin reads its
  `.name`; it never writes the document, and never puts the path in a result
  (FM-50).
- **`dry_run`** is `microscope.dry_run`, for a plugin that must label what
  it did not do.
- **`progress`** builds a `Progress` and hands it to the runner's
  `on_progress`. Not validated (a cosmetic bug must not end a scan). A
  callback that raises an `Exception` is logged once at WARNING and not
  called again in this run; the run goes on. `KeyboardInterrupt` passes.
- **`should_stop`** is true once the runner's `stop` event is set (a UI's
  Stop, a test). The CLI has no stop event: Ctrl-C is the
  `KeyboardInterrupt` that the Executor already turns into a stop (M1 §9).
  **`raise_if_stopped`** raises `StopRequested` when it is true.
- **`wait_s`** waits up to `seconds` (finite, ≥ 0, else `ValueError`) on
  the stop event in slices of at most 0.1 s, and raises `StopRequested` if
  it is set. The slices keep Ctrl-C deliverable on Windows, where one long
  lock wait is not interrupted (CPython's documented behaviour, the cause
  of FM-83; not measured on a stand). A plugin never calls `time.sleep`.
- **`upstream(slot, model)`** returns the result of the slot's provider,
  validated by `model`, from the snapshot of the document the runner took
  before `run`. A slot not in `reads` raises `ValueError` (a plugin bug:
  the declaration is what the runner checked). A result that `model`
  refuses raises `UpstreamError` naming the slot, the provider and the
  first validation error, with the fix `run <provider> again`.
- **`upstream_or_none(slot, model)`** is the same for a slot in
  `reads_optional`; it returns `None` when the slot is not usable (§1.8),
  and a usable result that `model` refuses still raises `UpstreamError`.
- **`save_checkpoint(state)`** writes `state.model_dump(mode="json")` into
  the running record's `checkpoint` and the record to the run document
  (§1.7, atomically). A plugin calls it after each unit of work it would
  not want to repeat (a well, a field). A `RunDocumentError` from it ends
  the run.
- **`checkpoint(model)`** returns the checkpoint of the attempt being
  resumed, validated by `model`, or `None` when the run was not started
  with `resume=True` or the previous attempt saved none. A checkpoint that
  `model` refuses raises `ResumeError`.

### 1.5 The registry — `smc/plugins/registry.py`

```python
GROUP = "smc.plugins"


@dataclass(frozen=True, slots=True)
class PluginEntry:
    plugin: Plugin[Any, Any]
    target: str
    distribution: str | None
    version: str | None


@dataclass(frozen=True, slots=True)
class LoadFailure:
    name: str
    target: str
    distribution: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Registry:
    plugins: Mapping[str, PluginEntry]
    failures: tuple[LoadFailure, ...]


def discover(entry_points: Iterable[EntryPoint] | None = None) -> Registry: ...
def load(
    name: str, entry_points: Iterable[EntryPoint] | None = None
) -> PluginEntry: ...
```

`entry_points=None` reads `importlib.metadata.entry_points(group=GROUP)`;
tests pass their own `EntryPoint` objects (`value=f"{__name__}:Good"`
loads a class of the test module itself, §1.1). `target` is the entry
point's `value`; `distribution` and `version` come from `ep.dist` when it
is set.

**Failures are findings.** `discover()` never raises for a plugin: each
entry that fails one of these checks becomes a `LoadFailure` and the
others load. In order:

1. Identical `(name, value)` pairs are one entry (an editable install can
   list a distribution twice). Two entries with one name and different
   values are both failures, `registered twice: <target> (<dist>), <target>
   (<dist>)`: nothing shadows a plugin silently.
2. `ep.load()` raising any `Exception`: `could not be imported:
   <Type>: <message>`.
3. Not a class: `names a <type name>, not a class`. Instantiating it
   raising: `could not be created: <Type>: <message>`.
4. `isinstance(instance, Plugin)` false: `lacks <member>, <member>`, the
   missing members by name.
5. `name` not equal to the entry point's name, or not matching
   `NAME_PATTERN`; `provides` or a read not matching it; `reads` and
   `reads_optional` not disjoint, or `reads_optional` holding `provides`
   (`reads` may hold it since §24 D8: a tool that turns the slot it reads).
6. `Params` not a subclass of `PluginParams`, `Result` not one of
   `PluginResult`; `requires` not a tuple of classes that are keys of
   `CAPABILITY_ROLES`: `requires <name>, which is not a capability`.

`KeyboardInterrupt` and `SystemExit` raised by an import pass through.
`load(name)` imports only the entries with that name: an unknown name
raises `PluginNotFoundError` with the names of the group (read without
importing anything), and a failure raises `PluginLoadError` holding the
`LoadFailure`. `plugins` is ordered by name; `failures` by name, then
target.

### 1.6 Parameters from text — `smc/plugins/params.py`

```python
def parse_params(model: type[P], pairs: Sequence[str]) -> P: ...
def describe_params(model: type[PluginParams]) -> list[tuple[str, str, str, str]]: ...
```

`parse_params` is what `smc run --param` uses, and a future UI form posts
the same strings. Every problem is collected, then one `ParamsError`
lists them all:

1. Each item splits at its first `=`. No `=`, or an empty key: `expected
   key=value, got '<item>'`. The value may be empty, and may contain `=`.
2. A key that is not a field of `model`: `unknown key '<key>' (keys: a,
   b, c)`. A key given twice: `'<key>' given twice`. Keys are field names
   exactly (snake_case); there are no aliases.
3. The field's annotation, with `None` removed from a union, is
   *scalar* when it is `str`, `int`, `float`, `bool`, `Path`, an `Enum`
   subclass or a `Literal`. A scalar value is passed as the raw string
   (§1.1); the exact text `null` is `None` when the annotation admits
   `None`. Any other field takes JSON: the value goes through
   `json.loads`, and a decode error reads `'<key>' takes JSON, for example
   ["A1", "B2"]`.
4. `model.model_validate(raw)`: each validation error becomes
   `<key>: <pydantic message>`.

`ParamsError` says `smc run <plugin> --show-params` lists the keys.

`describe_params` gives one row per field, in declaration order: the name;
the type (`str(annotation)` with `typing.` removed, a class by its
`__name__`); the default (`repr`, `(required)` when there is none); the
constraints (`ge=1 le=100` from the field's `annotated_types` metadata,
`""` when none) followed by the field's `description`.

### 1.7 The run document — `smc/run/document.py`

One JSON file per experiment. A stage's record is keyed by the **plugin's
name**; its result fills the slot that the plugin `provides`.

```python
SCHEMA_VERSION = 1


class StageStatus(str, Enum):
    running = "running"
    done = "done"
    stopped = "stopped"
    failed = "failed"
```

```python
class ReadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin: str
    attempt: str
```

```python
class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin: str
    provides: str
    attempt: str
    status: StageStatus
    started_utc: AwareDatetime
    finished_utc: AwareDatetime | None = None
    profile: str
    dry_run: bool
    smc_version: str
    plugin_version: str | None = None
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    reads: dict[str, ReadRecord | None] = {}
    error: str | None = None
```

```python
class RunDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    created_utc: AwareDatetime
    stages: dict[str, StageRecord] = {}

    @classmethod
    def new(cls) -> RunDocument: ...
    @classmethod
    def load(cls, path: Path) -> RunDocument: ...
    @classmethod
    def load_or_new(cls, path: Path) -> RunDocument: ...
    def provider(self, slot: str) -> StageRecord | None: ...
    def is_stale(self, plugin: str) -> bool: ...
    def unusable(self, slot: str, *, profile: str, dry_run: bool) -> str | None: ...
```

```python
class RunDocumentError(Exception):
    path: Path
    reason: str
    kept: Path | None
    record: StageRecord | None


def write_stage(path: Path, record: StageRecord) -> RunDocument: ...
```

**Fields.**

- `attempt` is `uuid.uuid4().hex`, new for every run of a stage, resumed
  runs included. It is what a downstream read records.
- `profile` is `microscope.profile.microscope.name`; `smc_version` is
  `smc.__version__`; `plugin_version` is the distribution's version from
  the registry, `None` when unknown. No host name, user name or serial
  number goes into the document: run documents are attached to issues
  (FM-50).
- `params` and `result` are `model_dump(mode="json")`. The document is
  plain data: it loads, and `smc run --list` works, without the plugin
  that wrote it installed; a reader validates a result with the model it
  expects (§1.4).
- `reads` maps each slot of `reads` and `reads_optional` to the provider's
  plugin and attempt at the start of the run, `None` for an optional slot
  that was not usable.
- `error` is one line, `<Type>: <message>`, for `failed` and for a
  `stopped` that came from `KeyboardInterrupt` (`KeyboardInterrupt: Ctrl-C`).
- `checkpoint` is cleared when the stage ends `done`.

**Reading.** `load` reads the bytes and decodes them as `utf-8-sig` (a
byte-order mark from Notepad is accepted, FM-80). It raises
`RunDocumentError` (and never returns a partial document) for: a missing
file; a directory; an `OSError`; bytes that are not UTF-8; invalid JSON
(with line and column); a top level that is not an object; a
`schema_version` that is not an integer, below 1, or above
`SCHEMA_VERSION` (`written by a newer smc (schema N); this smc reads
schema 1 — upgrade smc`); any validation error (its location and
message). `load_or_new` returns `RunDocument.new()` for a missing file
only, and nothing is written until a stage is: **an unreadable document is
never overwritten**, it may be the only record of an experiment.
`extra="forbid"` at every level means a document with a field this smc
does not know is refused rather than loaded and rewritten without it
(FM-107). Every change of the document's fields therefore bumps
`SCHEMA_VERSION`, and `load` migrates an older version in code.

**Writing.** `write_stage(path, record)` loads the file fresh
(`load_or_new`), sets `stages[record.plugin] = record`, and replaces the
file atomically; it returns the document written. It never writes from a
copy held in memory, so a second writer's other stages survive. Two writes
of the same file at the same instant can still lose one record: one writer
per run document is assumed and not locked (§1.13). The bytes are
`json.dumps(doc.model_dump(mode="json"), indent=2, ensure_ascii=True,
allow_nan=False) + "\n"`, ASCII only: a lone surrogate (FM-05) and `µ`
are escaped, so no console or code page can mangle the file. The
replacement follows `backends/mm.py::_write_atomically` (FM-38): a
temporary `.<name>.<token>.smc-tmp` beside the file, created with
`O_EXCL` and mode `0o666`, written, `fsync`ed, closed, then `os.replace`,
retried 5 times 0.05 s apart on `PermissionError` (antivirus or an indexer
holding the file on Windows, FM-24, as `_mm_child._write` does). A failed
write removes the temporary file and raises `RunDocumentError` saying the
document is as it was; a failed rename keeps the temporary file, names it
in `kept`, and says so. The helper is private to `document.py` (the five
existing copies are not merged in this issue).

**Slots.** `provider(slot)` is the record with the latest `started_utc`
among those whose `provides` is `slot` (ties broken by plugin name), `None`
when there is none. The latest attempt decides even when it did not finish:
a re-run of plate registration that failed leaves no usable plate, rather
than the older one silently. A **read never resolves to the reader's own
record** (§24 D8): `provider(slot, excluding=reader)` leaves it out, and
`is_stale`, `unusable(..., reader=)`, the runner and `RunContext` judge a
plugin's reads that way, which changes nothing for a plugin that does not
read the slot it provides.

**Staleness.** `is_stale(plugin)` is true when the plugin's record read a
slot whose provider has changed since: for each `(slot, read)` in its
`reads`, the current `provider(slot)` is `None` where `read` was not, or
another plugin or attempt than `read`, or is itself stale (recursively,
with a visited set). An optional slot read as `None` that now has a `done`
provider also makes the record stale. A record that never read anything is
never stale.

**Usability.** `unusable(slot, *, profile, dry_run)` returns `None` when
the slot can be read, else the reason, each with its fix:

| Condition | Reason |
|---|---|
| no provider | `the run document has no <slot>; smc run --list shows which plugin provides it` |
| provider not `done` | `the last run of <plugin> ended <status>; run it again` (`running` reads `is still running or was interrupted`) |
| provider stale | `<plugin> read its inputs from an earlier run; run it again` |
| provider's `profile` differs | `<plugin> was run on <profile>, this is <profile>` |
| provider was a dry run and this run is not | `<plugin> was a dry run; run it without --dry-run` |

A dry run may read a real result.

### 1.8 The runner — `smc/plugins/runner.py`

```python
def check_ready(
    plugin: Plugin[Any, Any],
    doc: RunDocument,
    params: PluginParams,
    *,
    profile: str,
    dry_run: bool,
    resume: bool,
) -> None: ...
```

```python
def run_plugin(
    plugin: Plugin[Any, Any],
    microscope: Microscope,
    params: PluginParams,
    run_doc: Path,
    *,
    resume: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
    stop: threading.Event | None = None,
    plugin_version: str | None = None,
) -> StageRecord: ...
```

`check_ready` touches neither hardware nor files, so the CLI calls it
before it opens the stand (FM-30), and `run_plugin` calls it again on the
document it loads. It raises `UpstreamError` for the first slot of
`reads` that is `unusable`, with that reason; and, when `resume` is true,
`ResumeError` when the plugin has no record (`nothing to resume`), its
record is `done` (`<plugin> finished; run it without --resume`), its
`params` differ from `params.model_dump(mode="json")` (`the parameters
differ from the run being resumed: <keys>`), or its profile differs.

`run_plugin`, in this order:

1. `params` not an instance of `plugin.Params`: `TypeError` (a bug).
2. `doc = RunDocument.load_or_new(run_doc)`; `check_ready(...)` with
   `profile=microscope.profile.microscope.name` and
   `dry_run=microscope.dry_run`.
3. Requirements: every capability of `requires` for which
   `microscope.has()` is false. When any is missing, `RequirementsError`
   lists them all, with each `CapabilityMissingError` message that
   `microscope.require()` gives. Nothing has been written.
4. The `running` record: a new `attempt`, `started_utc` now, `reads` from
   the document (the optional slots that are `unusable` recorded as `None`
   and logged at INFO with their reason), and, when resuming, the previous
   record's `checkpoint`. `write_stage`. A `RunDocumentError` here ends the
   run before any hardware is touched. When the plugin has a `stopped` or
   `failed` record with a checkpoint and `resume` is false, the runner logs
   at INFO that `--resume` would continue it, and starts afresh.
5. `result = plugin.run(microscope, params, ctx)`, where `ctx` reads the
   snapshot `doc` of step 2.
6. The outcome, written with `write_stage`:
   - a `plugin.Result` instance: `done`, `result` dumped, `checkpoint`
     cleared, `finished_utc` set; the record is returned. Anything else:
     `failed` with `TypeError: <plugin> returned <type>, not <Result>`,
     and that `TypeError` raised.
   - `StopRequested`: `stopped`, `checkpoint` kept; the record is
     returned (a stop is an outcome, not an error).
   - `KeyboardInterrupt`: `stopped`, `error` set, `checkpoint` kept; then
     re-raised (FM-71: the operator's stop is not turned into a result).
   - any other `BaseException`: `failed`, `error` its one line,
     `checkpoint` kept; re-raised unchanged, so a `HardwareError` or a
     `SafetyRefusedError` reaches the CLI's mapping as it is.
   If the outcome's write fails while an exception is in flight, the
   write's error is logged at ERROR and the original exception is raised
   (the document then shows `running`, which `unusable` reads as
   interrupted). If the `done` write fails, `RunDocumentError` is raised
   with `record` set to the `done` record, so the caller can still show
   the result (FM-34).

The runner keeps no exception in a local or an attribute beyond its
`except` block (FM-98), and `RunContext` holds no reference to the
microscope (FM-92).

### 1.9 Errors — `smc/plugins/errors.py`

```python
class PluginError(Exception):
    """A plugin could not be found, loaded, configured or started."""
```

| Class | Attributes | Raised by | `smc run` exit |
|---|---|---|---|
| `PluginNotFoundError` | `name`, `available: tuple[str, ...]` | `load` | 2 |
| `PluginLoadError` | `failure: LoadFailure` | `load` | 1 |
| `ParamsError` | `plugin`, `problems: tuple[str, ...]` | `parse_params` | 2 |
| `RequirementsError` | `plugin`, `missing: tuple[str, ...]`, `details: tuple[str, ...]` | `run_plugin` | 1 |
| `UpstreamError` | `plugin`, `slot`, `reason` | `check_ready`, `ctx.upstream` | 2 |
| `ResumeError` | `plugin`, `reason` | `check_ready`, `ctx.checkpoint` | 2 |

All derive from `PluginError`, which is neither a `RuntimeError` nor a
`HardwareError`: a configuration problem is not a device fault.
`RunDocumentError` (in `smc.run`, exit 1) derives from `Exception` only.
Each class with its own `__init__` defines `__reduce__`, as
`smc.hardware.errors` does, so it survives a worker process. Messages name
the plugin and end with what to do (`smc run --list`, `--show-params`,
`smc run <provider> --run-doc <path>`), never only what went wrong.

The exit codes follow M1 §9: 2 when nothing was sent because of an
argument or a precondition the operator fixes with a command; 1 for a
hardware, profile or file error, a plugin that does not load, a stand that
lacks a capability (as `smc stage get` on a stand without one), and a
plugin's own exception; 130 for Ctrl-C.

### 1.10 CLI — `smc run`

```
smc run --list
smc run NAME --show-params
smc run NAME [-p PROFILE] --run-doc PATH [--param KEY=VALUE]... [--dry-run] [--resume]
```

One command, `@app.command("run")`, with `NAME` an optional argument.
`--list` with a `NAME`, or neither, is a usage error (exit 2).

- **`--list`** opens no stand. One aligned line per plugin, by name:
  `name  summary  requires XYStage, ZStage  provides <slot>  reads <slots>`
  (`-` for none; optional reads in parentheses), then one `✗ <name>
  (<target>): <reason>` line per `LoadFailure`. Exit 0, failures included:
  they are findings.
- **`--show-params`** opens no stand: `describe_params` as aligned rows,
  `none` when the model has no field. Exit 0.
- **A run**, in this order, so that nothing reaches the stand before
  every check that needs none has passed (FM-30). Steps 1 to 5 run inside
  one `with _Reported(ctx, after_report=...)`, so every error below gets
  its one line and its exit code; only step 4 holds the stand open.
  1. `load(NAME)`; `parse_params(entry.plugin.Params, params)`.
  2. `--run-doc` given (else usage error, exit 2: a run document is
     never created where the operator did not say); not a directory; its
     folder exists and is a directory (exit 2, like `smc snap --out`).
     `Path("")` is `.`, a directory, so an empty value is refused (FM-78).
  3. `Profile.load(profile)`; `RunDocument.load_or_new(run_doc)`;
     `check_ready(..., profile=<profile>.microscope.name, ...)`. A document
     that does not load exits 1 and is left as it is.
  4. `with Microscope.open(profile_obj, dry_run=dry_run) as microscope:`
     then `run_plugin(..., on_progress=..., plugin_version=entry.version)`.
     The stand is released before any `✗` line (M1 §9), and the result of
     step 5 is printed after it is released.
  5. Done: `✓ <name> done in <s:.1f> s — <run_doc>, stage <name>`, then the
     result as `json.dumps(record.result, indent=2, ensure_ascii=False)`
     through `_say`, then one `! <plugin> read <slot> from an earlier run;
     run it again` line for each stage that this run made stale (stale
     after, not before). Exit 0.
  6. Stopped by `StopRequested` (a plugin's own stop): `✗ <name> stopped`,
     exit 130.
- **Errors** go through `_report`, which gains two branches before its
  fallback: `PluginNotFoundError`, `ParamsError`, `UpstreamError`,
  `ResumeError` print `str(exc)` and return 2; any other `PluginError` and
  `RunDocumentError` print `str(exc)` and return 1. When a
  `RunDocumentError` carries a `record` with a result, `after_report`
  prints that result as in step 5, so the work is not lost (FM-34).
  Ctrl-C keeps `_report`'s line and exit 130; the record says `stopped`.
- **Progress** goes to stdout through `_say`, plain text:
  `<name>: <done>/<total> <message>` (`<name>: <done> <message>` without a
  total). The first call is printed, then a call at least 1 s after the
  last printed one, and every call with `done == total`, so a fast loop
  prints its first and last lines and a long one a line a second.
- Everything is plain text: no markup (FM-36, the `[roles.assign]` of a
  `RequirementsError` survives), no wrapping (FM-37), and the cp1252
  handling of `main` applies (FM-20).

### 1.11 The built-in plugin — `smc/plugins/stage_report.py`

`stage-report` reads where the stage is, so that the whole path (entry
point, parameters, requirements, progress, stop, run document, CLI) is
exercised on the demo devices without moving anything.

```python
class StageReportParams(PluginParams):
    samples: int = Field(1, ge=1, le=100, description="How many readings to take.")
    interval_ms: float = Field(
        0.0, ge=0.0, le=10_000.0, description="Wait between two readings."
    )
    include_z: bool = Field(
        True, description="Also read Z when there is a focus stage."
    )
```

```python
class StageReading(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    x_um: float
    y_um: float
    z_um: float | None
```

```python
class StageReportResult(PluginResult):
    profile: str
    readings: tuple[StageReading, ...]
```

`StageReport`: `name = provides = "stage-report"`, `requires = (XYStage,)`,
`reads = reads_optional = ()`. `run` takes `microscope.get(ZStage)` when
`include_z` (else `None`); then, for each sample, `ctx.raise_if_stopped()`,
one reading (`z_um` is `None` without a focus stage), `ctx.progress(i + 1,
samples, "read")`, and `ctx.wait_s(interval_ms / 1000)` between two
readings (not after the last). Registered in `pyproject.toml`:

```toml
[project.entry-points."smc.plugins"]
stage-report = "smc.plugins.stage_report:StageReport"
```

### 1.12 Tests

Plugins in tests are classes of the test module, loaded through
`EntryPoint(value=f"{__name__}:<Class>")`. Runner tests run on
`fake_microscope` (no Micro-Manager); the demo test and the CLI tests are
marked `demo`. Every run document lives under `tmp_path`.

### 1.13 Not decided here

- **#19's guided steps** (seed, go to a well, capture an edge, calibrate)
  span several invocations and need state between them. #19's section
  decides how (a checkpoint, or a new optional record field with a schema
  bump); §1 does not preclude either. *Decided in §5*: neither; the
  references are parameters.
- **Several writers of one run document** (a UI and a CLI on one
  experiment): `write_stage` re-reads before it writes, and nothing locks
  the file. A lock comes with the UI (M5) if it is needed.
- **Detector plugins** (#23) are a second family behind their own
  protocol; their discovery is #23's. *Decided in §9*: a `Detector`
  protocol, and no discovery in M3.
- **A pipeline command** that runs several stages in order is not in M3:
  each stage is run alone, and the document says what is missing.
- **JSON output** of `smc run` beyond the result (M1 §9: later).

## 2. The plate-scan workflow (#192)

### 2.1 What was measured (2026-10-27)

Measured on the demo devices (pymmcore-plus 0.18.1, useq-schema 0.9.2,
pydantic 2.13.5, Python 3.13), on the model, and read from
`nikon-control@1aad7c9` by subagents (scripts in the design session's
scratchpad, not committed):

- **useq.** The registry holds `6-`, `12-`, `24-`, `48-`, `96-`, `384-`
  and `1536-well` and two coverslips. `WellPlatePlan` keeps the order of
  `selected_wells`: `([1, 0, 0], [0, 5, 0])` gives `B1`, `A6`, `A1`; its
  `selected_well_names` are `numpy.str_`, so code converts them with
  `str()`. A `GridRowsColumns` grid is laid in **stage axes**: with a plate
  rotated by 0.5°, the four fields of a 2 × 2 grid in `A3` share their `y`
  two by two, while the well centres follow the rotation. Inside a well
  the fields are snake-ordered from the top left. Under `mypy --strict`,
  `WellPlatePlan(plate=...)` wants a `WellPlate`
  (`useq.WellPlate.from_str(name)`, which pydantic would otherwise do from
  the string) and `GridRowsColumns(overlap=...)` a pair of percentages.
  The forward model is M1 §8's: `world = A1 + R(+θ) · (col · pitch_x,
  −row · pitch_y)`, θ in degrees, pitches in mm in `WellPlate`.
- **pydantic.** A model holding a `useq.WellPlatePlan`, nested tuples
  (a 2 × 2 matrix as `tuple[tuple[float, float], tuple[float, float]]`)
  and `dict[str, int]` round-trips through `model_dump(mode="json")` and
  `model_validate`; a `field_validator` calling `WellPlate.from_str`
  refuses `"glass"` (`value_error`). **The dump of a subclass that adds a
  field is refused when read through its base** (`extra_forbidden`): a
  second provider of a slot cannot return a subclass of the slot's model
  (§2.3). The prototype of §3's models passes `mypy --strict` with the
  repository's configuration.
- **The camera matrix** of §3.2 was checked against
  `smc.testing.synthetic`: on a 256 × 320 frame, the content shifts that
  two 20 µm stage moves produce, measured by FFT cross-correlation, give
  back the sample's pixel size, rotation and mirror (0° and 30° upright,
  30° and −12° mirrored, at 1.0 and 0.5 µm per pixel), and a blob placed
  at (30, 10) µm maps back to it within a pixel.
- **The demo**: camera 512 × 512, `uint16`, 1.0 µm per pixel (`Res10x`;
  `Res20x` 0.5, `Res40x` 0.25), a 5 mm XY move takes 0.59 s, a snap
  12 ms; objectives `Nikon 40X Plan Fluor ELWD`, `Nikon 10X S Fluor`, …;
  an `Autofocus` device exists.
- **The Ti2-E** (`docs/hardware/nikon-ti2.md`, not measured here): a
  Prime 95B of 1412 × 1412 pixels of 11 µm (datasheet), so 0.275 µm per
  pixel at 40× is **assumed** and a field is 388 µm. A 96-well well
  (6.4 mm) then takes a 19 × 19 grid at 10 % overlap, 361 fields, of which
  **305 touch the well disc** (computed). A snap costs about 100 ms at
  10 ms of exposure (readout 29 ms, clear 58 ms).
- **The model** `local/models/cell_classes_model-cat-v2.pth` (166 MB):
  keys `model_state_dict`, `classes`, `epoch`, `val_mAP@0.5`, `init_from`,
  `trainable_layers`; classes `['__background__', 'single', 'doublet',
  'debris']`; epoch 6; `val_mAP@0.5` 0.867. It loads with
  `torch.load(weights_only=True)` into torchvision's
  `fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
  num_classes=4)` (torch 2.13, torchvision 0.28). The model resizes every
  frame to `min_size=800` (`max_size=1333`) itself, so the frame size is
  free: on an Apple M5 Pro CPU (6 threads), one frame costs **0.34 s at
  1024² and 0.38 s at 2048²**, normalisation included; MPS 0.08 s. Noise
  frames gave no detection. The microscope PC's CPU is not measured
  (#198).
- **What `nikon-control` has and has not.** It has the camera matrix
  (`camera_cal.py`), plate registration and lattice fit (`plate.py`,
  `plate_find.py`), well selection (`wells.py`), the scan loop (`scan.py`),
  the timing model (`timing.py`) and the in-frame suppression of
  overlapping boxes (`overlap.py`). It has **no** de-duplication across
  fields, **no** clipping of a grid to a round well, **no** choice of
  fields by cell count and **no** route (2-opt is a line in
  `docs/acquisition-architecture.md`). Those are new design here, not
  ports. Its `/scope` page is plain Bokeh, not Panel.

### 2.2 The stages and their slots

```text
camera-calibration ─────────────(optional)──────────┐
plate-registration ─┐                                │
plate-walls ────────┤                                │
plate-finding ──────┴─ plate ──► scan-plan ◄─────────┘ ◄── timings (optional)
                                    │
                                   scan ──► cells ──► density ──┐
                                    │         │                 │
                                    │         └────► route ◄────┼── timings (optional)
                                    │                  │        │
                                    └──────────────────┴────────┴─► report
                                                       └──────────► nis-export
```

| Plugin | Slot it fills | `reads` | `reads_optional` | `requires` | Issue | § |
|---|---|---|---|---|---|---|
| `camera-calibration` | `camera-calibration` | — | — | `XYStage`, `Camera` | #22 | 4 |
| `plate-registration` | `plate` | — | — | — | #19 | 5 |
| `goto-well` | `goto-well` | `plate` | — | `XYStage` | #19 | 5 |
| `plate-finding` | `plate` | — | — | `XYStage`, `Camera` | #20 | 6 |
| `plate-walls` | `plate` | `camera-calibration` | — | `XYStage`, `Camera` | #328 | 17 |
| `scan-plan` | `scan-plan` | `plate` | `camera-calibration`, `timings` | `Camera` | #21 | 7 |
| `scan` | `scan` | `scan-plan` | — | `XYStage`, `Camera` | #213 | 8 |
| `cells` | `cells` | `scan` | — | — | #23 | 9 |
| `density` | `density` | `scan-plan`, `scan`, `cells` | — | — | #193 | 10 |
| `route` | `route` | `scan-plan`, `scan`, `cells` | `timings` | — | #194 | 11 |
| `timing` | `timings` | — | — | `XYStage`, `Camera` | #24 | 12 |
| `report` | `report` | `scan-plan`, `scan`, `cells`, `density` | `route` | — | #195 | 13 |
| `nis-export` | `nis-export` | `route` | — | — | #196 | 14 |

The plate page (#197, §15) runs these plugins; it is not one.

Each stage is run alone, by `smc run <name>` or by the page, and the run
document says what is missing (§1.13: no pipeline command). A stage that
re-runs makes its readers stale (§1.7), so registering the plate again
after a scan makes the scan, the cells, the density, the route and the
report stale, and `smc run` says so.

### 2.3 Rules every stage follows

- **The slot models live in one package, `smc.plugins.slots` (§3),
  written first (#199).** This overrides §1.3's "the model of a slot
  belongs to the plugin that introduces it" for the workflow: plugins
  built in parallel would otherwise import modules that do not exist yet.
  A plugin imports its `Result` from there and defines only its `Params`
  and its checkpoint model.
- **A second provider of a slot returns the slot's model itself**, never a
  subclass (§2.1: a subclass's dump is refused through the base).
  `plate-finding` fills the optional `fit` of `PlateResult`.
- **A stage that ran and could not produce a usable result raises
  `PluginRunError` (§3.6)**: the record ends `failed` with the reason, its
  checkpoint is kept, and no unusable result fills the slot for a reader
  to trust. A result that is `done` is usable; it carries what it assumed
  in `assumed` (§1.3).
- **Files** a stage writes go under `run_dir / <plugin name> / <token>`
  (or `<token>.<suffix>`), where the token is new for every run of the
  stage (`smc.run.files.new_token()`) and kept across a resume through the
  checkpoint. A file that a `done` result names is never rewritten: a new
  run writes a new token. Paths in results are relative to `run_dir`,
  POSIX (FM-24), and read back through `smc.run.files.resolve_in_run`,
  which refuses one that leaves the folder.
- **No stage turns the objective turret** (FM-14). A stage that needs
  another objective refuses and says which one to put in place.
- **No stage moves Z.** Focus during the scan is the autofocus's (§8).
- **A loop between two stage moves** (camera calibration, plate finding,
  the scan, timing) calls `ctx.raise_if_stopped()` before each move,
  waits only through `ctx.wait_s`, and catches `SafetyRefusedError` and
  `HardwareError` only where this document says so, always letting
  `MicroscopeHaltedError` and `MotionStoppedError` through first (FM-64:
  the operator's stop ends the stage, it is not a failed field).
- **A return to the start position runs only after the measurement
  returned normally** (FM-88). After a stop, a halt or a fault the stage is
  left where it is, and a WARNING says so.
- **Dependencies.** `scikit-image>=0.22` becomes a core dependency (#22,
  phase correlation): a built-in plugin whose import fails would be a
  `LoadFailure` on every PC. torch is the optional extra `detect`
  (`torch>=2.2`, `torchvision>=0.17`, #23), imported only inside
  `TorchDetector`. Panel is the optional extra `ui` (`panel>=1.4`, #197),
  imported only by `smc.ui.plate_page` and the `smc ui` command. Versions
  of the extras are not measured on the microscope PC.

## 3. The foundation — slot models, geometry, files (#199)

```
src/smc/plugins/slots/
  __init__.py            re-exports every name below
  base.py                SlotValue
  plate.py               WellRef, WallTouch, LatticeFit, PlateResult, WellGeometry,
                         well_geometry(), normalise_well()
  camera_calibration.py  CameraCalibration
  scan_plan.py           PlannedField, ScanPlanResult
  scan.py                FocusState, ScanFrame, FailedField, ScanResult, load_frame()
  cells.py               Cell, CellsFile, CellsResult, load_cells()
  density.py             DensityWell, DensityResult
  route.py               RoutePosition, RouteResult
  timings.py             TimingsResult
  report.py              ReportResult
  nis_export.py          NisExportResult
src/smc/plugins/geometry.py   Matrix2, pixel_matrix(), stage_of_pixel(), pixel_of_stage(), footprint()
src/smc/plugins/errors.py     + PluginRunError
src/smc/run/files.py          AtomicWriteError, write_atomically(), retry_on_permission_error(),
                              new_token(), relative_to_run(), resolve_in_run()
```

Every model below derives from `PluginResult` (a slot's result) or from
`SlotValue`, the base for values inside a result. It is defined in
`slots/base.py` and re-exported by the package, so `from smc.plugins.slots
import SlotValue` holds: defined in `__init__.py`, below the imports of the
modules that need it, it fails with a circular `ImportError` (measured), and
above them `ruff`'s `I` and `E402` refuse it.

```python
class SlotValue(BaseModel):
    """A value inside a slot's result: frozen, strict about keys, finite."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
```

### 3.1 Geometry — `smc/plugins/geometry.py`

```python
Matrix2 = tuple[tuple[float, float], tuple[float, float]]


def pixel_matrix(pixel_size_um: float) -> Matrix2: ...
def stage_of_pixel(
    x_um: float,
    y_um: float,
    col_px: float,
    row_px: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> tuple[float, float]: ...
def pixel_of_stage(
    x_um: float,
    y_um: float,
    px_x_um: float,
    px_y_um: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> tuple[float, float]: ...
def footprint(
    x_um: float, y_um: float, *, matrix: Matrix2, image_shape_px: tuple[int, int]
) -> tuple[tuple[float, float], ...]: ...
def fov_of(matrix: Matrix2, image_shape_px: tuple[int, int]) -> tuple[float, float]: ...
```

- **The matrix** `M` maps a pixel offset `(d_col, d_row)` from the frame
  centre to a stage offset `(dx_um, dy_um)`: `d = M · (d_col, d_row)`.
  Rows grow downwards, stage Y upwards (M1 §2), so an upright camera
  without mirror has `M = ps · R(θ) · diag(1, −1)`; a mirrored one
  `M = ps · R(θ) · diag(−1, 1) · diag(1, −1)`. This is the convention of
  `smc.testing.synthetic` (`camera_rotation_deg`, `mirrored`), checked in
  §2.1. `nikon-control` called the upright camera "mirrored" (its test: "a
  right-handed mapping is 'mirrored' in image terms"); here it is not.
- **The frame centre** is `((W − 1) / 2, (H − 1) / 2)` in `(col, row)`,
  as in the synthetic sample (`nikon-control` used `W / 2`: half a pixel
  off). The stage position recorded with a frame is the point imaged at
  that centre.
- `pixel_matrix(ps)` is `((ps, 0.0), (0.0, -ps))`: the matrix assumed when
  no camera calibration exists (rotation 0, no mirror). It raises
  `ValueError` for a pixel size that is not finite and positive (FM-104).
- `stage_of_pixel` gives the stage point imaged at pixel `(col_px,
  row_px)` of a frame taken at `(x_um, y_um)`; `pixel_of_stage` the
  inverse (pixel `(col, row)` of a stage point `(px_x_um, px_y_um)`).
  `pixel_of_stage` raises `ValueError` for a singular matrix
  (`abs(det) < 1e-12`).
- `footprint` gives the four stage corners of the frame, in the order of
  the pixel corners `(0, 0)`, `(W, 0)`, `(W, H)`, `(0, H)` (the outer edges
  of the edge pixels, `−0.5` and `W − 0.5`), a rotated rectangle.
- `fov_of` gives `(width, height)` of the frame's footprint in stage axes,
  in µm: the extent of the four corners of `footprint(0, 0, ...)`, so for a
  camera turned by the stage the bounding box of the rotated frame. It was
  defined twice, in `scan_plan.algorithm` (the grid steps by it) and
  `route.algorithm` (the candidate lattice and its counting window use it),
  and lives here since #232 (the footprints themselves stay rotated
  rectangles; only the extent is a box).

### 3.2 Plate — `slots/plate.py` (slot `plate`)

```python
class WellRef(SlotValue):
    well: str
    x_um: float
    y_um: float
```

```python
class WallTouch(SlotValue):
    well: str
    side: WallSide  # "+X", "+Y", "-X", "-Y": a stage direction, §19 (was left, ...)
    x_um: float
    y_um: float
```

```python
class LatticeFit(SlotValue):
    score: float = Field(ge=0.0, le=1.0)
    polarity: Literal[1, -1]
    runner_up: str | None
    margin: float | None
    n_samples: int = Field(ge=0)
    n_inside: int = Field(ge=0)
    notes: tuple[str, ...] = ()
```

```python
class PlateResult(PluginResult):
    plate: str
    a1_center_xy_um: tuple[float, float]
    rotation_deg: float = Field(ge=-180.0, le=180.0)
    method: Literal["seed", "references", "lattice-fit", "wall-images"]
    residual_um: float | None = Field(ge=0.0)
    scale_error: float | None
    n_refs: int = Field(ge=0)
    refs: tuple[WellRef, ...] = ()
    fit: LatticeFit | None = None
    walls: WallImages | None = None

    def to_plan(self, **kwargs: Any) -> useq.WellPlatePlan: ...
    def well_center_um(self, well: str) -> tuple[float, float]: ...
```

`method="wall-images"` and `walls` (the model is §17.2's) are #328's: the
plate fitted to the centres of wells measured from images of their walls.
Both are additive; a run document written before them validates unchanged.

- `plate` is validated by `useq.WellPlate.from_str` (an unknown name is a
  validation error). Well names are useq's (`A1`, `H12`, `AA3`), always
  normalised by `normalise_well` (below) before they reach a model.
- `residual_um` is the worst distance between a reference and the fitted
  well centre; `None` when nothing measured it (a seed, a lattice fit).
  `scale_error` is the measured spacing over the nominal, minus 1; `None`
  with fewer than two references.
- `to_plan(**kwargs)` is `useq.WellPlatePlan(plate=WellPlate.from_str(...),
  a1_center_xy=..., rotation=..., **kwargs)`. `well_center_um` raises
  `ValueError` for a well that is not on the plate.

```python
@dataclass(frozen=True, slots=True)
class WellGeometry:
    well: str
    center_um: tuple[float, float]
    size_um: tuple[float, float]
    circular: bool
    rotation_deg: float

    def contains(self, x_um: float, y_um: float, *, margin_um: float = 0.0) -> bool: ...
    def contains_many(
        self,
        x_um: npt.NDArray[np.float64],
        y_um: npt.NDArray[np.float64],
        *,
        margin_um: float = 0.0,
    ) -> npt.NDArray[np.bool_]: ...
    def to_plate_frame(self, x_um: float, y_um: float) -> tuple[float, float]: ...


def well_geometry(plan: useq.WellPlatePlan, well: str) -> WellGeometry: ...
def normalise_well(name: str) -> str: ...
```

`size_um` is `(width, height)` from `plate.well_size` in mm × 1000; the
centre comes from the forward model of §2.1 (it agrees with useq's own
`all_well_positions` to 3e-11 µm on all nine registered plates, rotated by
13°), and `rotation_deg` is `plan.rotation`, which useq leaves `None` on a plan
built without one: it is read as 0.0 (measured, useq 0.9.2). The row and
column are found in `plan.all_well_names`, never by parsing the letters. All
registered round wells have equal sizes (measured). `to_plate_frame` gives the
offset from the centre rotated by `−rotation_deg`, in which a square well
is axis-aligned. `contains` is true inside the well shrunk by
`margin_um`: a disc of radius `width / 2 − margin` (the registry's round
wells have equal sizes), or the rectangle of half-sizes minus the margin.
Density, the route, the report and the scan plan's clipping use it, and
nothing else re-derives a well's shape.

`contains_many` is `contains` over arrays (added for #193, which tests a
grid of a million points per well): `x_um` and `y_um` broadcast against each
other, so a row and a column give a grid, and the result has the broadcast
shape. It is the same arithmetic in the same order, so it agrees with
`contains` on every point, the edge and the margin's refusal included (FM-132;
125 200 points compared, none differing, measured 2026-10-28, edges
exactly on the boundary among them); a point that is not finite, or so far
that its square overflows, is outside, with no numpy warning (`np.errstate`
inside: Python's floats go to `inf` silently, numpy's warn, and `-W error`
turns that into a failure). Nothing else knows the shape either: the
rasterised area of §10 asks it, and does not re-derive it.

### 3.3 Camera calibration — `slots/camera_calibration.py` (slot `camera-calibration`)

```python
class CameraCalibration(PluginResult):
    objective: str | None
    image_shape_px: tuple[int, int]
    matrix_um_per_px: Matrix2
    pixel_size_um: float = Field(gt=0.0)
    rotation_deg: float = Field(ge=-180.0, le=180.0)
    mirrored: bool
    residual_um: float = Field(ge=0.0)
    response: float = Field(ge=0.0, le=1.0)
    return_error_um: float | None = Field(ge=0.0)
    step_um: float = Field(gt=0.0)
    reused_from: str | None = None
```

`image_shape_px` is `(height, width)` at the time of the measurement (a
binning change makes the calibration inapplicable, §7). `pixel_size_um` is
`sqrt(|det M|)`; `rotation_deg` and `mirrored` follow §3.1;
`residual_um` and `response` are §4's; `return_error_um` is the distance,
in µm, between the start and the content after the return (§4), `None`
when the measurement was reused or the frame after the return did not
correlate (unknown, not zero).

### 3.4 Scan plan and scan — `slots/scan_plan.py`, `slots/scan.py`

```python
class PlannedField(SlotValue):
    well: str
    index: int = Field(ge=0)
    name: str
    x_um: float
    y_um: float
```

```python
class ScanPlanResult(PluginResult):
    plate_plan: useq.WellPlatePlan
    wells: tuple[str, ...]
    objective: str | None
    pixel_size_um: float = Field(gt=0.0)
    matrix_um_per_px: Matrix2
    matrix_source: Literal["camera-calibration", "profile", "parameter"]
    image_shape_px: tuple[int, int]
    fov_um: tuple[float, float]
    overlap: float = Field(ge=0.0, lt=1.0)
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    fields: tuple[PlannedField, ...]
    estimate_s: float | None = Field(ge=0.0)
```

`plate_plan` holds the plate, its registration, the selected wells in
visiting order and the `GridRowsColumns`; `fields` are its image positions
that survived clipping (§7), in visiting order, `index` counting from 0 in
each well, `name` useq's (`A1_0003`, gaps included). `fov_um` is `(width,
height)`, the footprint's extent in stage axes.

```python
FocusState = Literal["locked", "unlocked", "off", "none"]
```

```python
class ScanFrame(SlotValue):
    well: str
    index: int = Field(ge=0)
    name: str
    file: str
    x_um: float
    y_um: float
    z_um: float | None
    focus: FocusState
    pfs_offset: float | None
    t_s: float = Field(ge=0.0)
    # Added by #341 and #430, last and optional (§21.5, §8 *Refined while planning #430*):
    cell_focus: FieldFocus | None = None
    well_dx_um: float | None = None
    well_dy_um: float | None = None
```

```python
class FailedField(SlotValue):
    well: str
    index: int = Field(ge=0)
    name: str
    x_um: float
    y_um: float
    error: str
```

```python
class ScanResult(PluginResult):
    folder: str
    objective: str | None
    pixel_size_um: float = Field(gt=0.0)
    matrix_um_per_px: Matrix2
    image_shape_px: tuple[int, int]
    exposure_ms: float = Field(ge=0.0)
    frames: tuple[ScanFrame, ...]
    failed: tuple[FailedField, ...]

    def frame_index(self) -> dict[str, list[int]]: ...


def load_frame(run_dir: Path, frame: ScanFrame) -> npt.NDArray[Any]: ...
```

- A frame's `x_um`, `y_um` are the stage's **readback** after the move,
  not the target (the demo rounds a command, M1 §8); `FailedField` keeps
  the target. `z_um` is the focus stage's reading, `None` without one;
  `pfs_offset` the autofocus offset in its native unit, `None` when the
  autofocus was not used. `focus`: `locked` (used and locked when the
  frame was taken), `unlocked` (used, not locked), `off` (the stand has
  an autofocus the scan did not use), `none` (it has none). `t_s` counts
  from the start of the attempt that took the frame. `well_dx_um`,
  `well_dy_um`: where the frame is in its well, the readback minus the
  well's centre in stage axes (§8, *Refined while planning #430*); `None`,
  both, in a scan written before they existed.
- `matrix_um_per_px` and `pixel_size_um` are the scan plan's, copied so
  that the cells need only `scan`. `frame_index()` maps a well to the
  indices of its frames in `frames`.
- `load_frame` reads `resolve_in_run(run_dir, frame.file)` with
  `tifffile.imread` and raises `FileNotFoundError` naming the file.

### 3.5 Cells, density, route, timings, report, export

```python
class Cell(SlotValue):
    well: str
    x_um: float
    y_um: float
    label: str
    score: float = Field(ge=0.0, le=1.0)
    frame: int = Field(ge=0)
    box_px: tuple[float, float, float, float]
```

```python
class CellsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = 1
    cells: tuple[Cell, ...]
```

```python
class CellsResult(PluginResult):
    file: str
    model: str
    model_sha256: str
    classes: tuple[str, ...]
    device: str
    score_threshold: float
    merge_radius_um: float
    frames: int = Field(ge=0)
    counts: dict[str, dict[str, int]]
    merged: int = Field(ge=0)
    seconds_per_frame: float | None = Field(ge=0.0)


def load_cells(run_dir: Path, result: CellsResult) -> tuple[Cell, ...]: ...
```

- **The issue's `(well, x_um, y_um, class, score, frame)`**: `class` is a
  Python keyword, so the field is `label`. `frame` is the index of the
  frame in `ScanResult.frames`; `box_px` is `(row0, col0, row1, col1)` in
  that frame's pixels, for the report's overlays. `x_um`, `y_um` are the
  centre of the box in stage µm (§3.1).
- **The cells live in a file**, not in the run document: a 96-well well
  at 40× is about 300 frames, and thousands of cells per well would make a
  document of megabytes that every checkpoint rewrites. `file` is
  `cells/<token>.json`, a `CellsFile` written with `ensure_ascii=True` as
  the document is (§1.7). `counts` (well → label → count, every label of
  `classes` present, zeros included) stays in the result, so a reader
  that needs only counts does not open the file.
- `load_cells` decodes the file as `utf-8-sig`, validates it, and raises
  `ValueError` naming the file when it is missing, unreadable, of another
  `schema_version`, or holds a count per well and label that differs from
  `result.counts` (a file replaced behind the result). Readers turn that
  into a `PluginRunError` that says to run `cells` again.
- `model` is the weights' file name (never its full path: FM-50, run
  documents are attached to issues), or `blobs` for the blob detector of
  §9.1; `model_sha256` its SHA-256 (of the detector's identity for
  `blobs`);
  `classes` the detector's, background excluded.

```python
class DensityWell(SlotValue):
    well: str
    fields: int = Field(ge=0)
    failed_fields: int = Field(ge=0)
    area_mm2: float = Field(ge=0.0)
    counts: dict[str, int]
    outside_well: int = Field(ge=0)
    per_mm2: dict[str, float] | None
```

```python
class DensityResult(PluginResult):
    labels: tuple[str, ...]
    area_grid_um: float = Field(gt=0.0)
    wells: tuple[DensityWell, ...]
```

`per_mm2` is `None` when `area_mm2` is 0 (no frame of the well was taken:
unknown, not zero).

```python
class RoutePosition(SlotValue):
    order: int = Field(ge=0)
    well: str
    name: str
    x_um: float
    y_um: float
    z_um: float | None
    pfs_offset: float | None
    counts: dict[str, int]
```

```python
class RouteResult(PluginResult):
    objective: str | None
    matrix_um_per_px: Matrix2
    image_shape_px: tuple[int, int]
    fov_um: tuple[float, float]
    labels: tuple[str, ...]
    positions: tuple[RoutePosition, ...]
    uncovered: dict[str, int]
    distance_um: float = Field(ge=0.0)
    travel_s: float | None = Field(ge=0.0)

    def to_useq(self) -> tuple[useq.AbsolutePosition, ...]: ...
```

`to_useq()` gives `AbsolutePosition(x=x_um, y=y_um, z=z_um, name=name)` in
route order: ADR-0004's "where to image is a useq object", derived from
the one stored list rather than stored twice. `uncovered` counts, per
well, the cells of the wanted labels that no chosen field holds.

```python
class TimingsResult(PluginResult):
    move_overhead_ms: float = Field(ge=0.0)
    move_ms_per_mm: float = Field(ge=0.0)
    refocus_ms: float | None = Field(ge=0.0)
    snap_ms: float = Field(ge=0.0)
    channel_ms: dict[str, float] = Field(default_factory=dict)
    channel_switch_ms: float | None = Field(None, ge=0.0)
    samples: int = Field(ge=0)

    def move_ms(self, distance_um: float) -> float: ...
```

`move_ms(d)` is `0` for `d <= 0`, else `move_overhead_ms + move_ms_per_mm
* d / 1000` (`timing.py`'s model). `refocus_ms` is `None` on a stand
without autofocus; a value that was assumed rather than measured is named
in `assumed`, as `timing.py` did.

```python
class ReportResult(PluginResult):
    file: str
```

```python
class NisExportResult(PluginResult):
    file: str
    positions: int = Field(ge=0)
    z_mode: Literal["z", "pfs-offset", "none"]
```

### 3.6 `PluginRunError` and the run folder's files

```python
class PluginRunError(PluginError):
    """A plugin ran and could not produce a usable result."""

    def __init__(self, plugin: str, reason: str) -> None: ...
```

Attributes `plugin`, `reason`; the message is `<plugin>: <reason>`, and
the reason ends with what to do. It defines `__reduce__` (§1.9). `smc run`
maps it to exit 1 through §1.10's "any other `PluginError`" branch, which
needs no change; §1.9's docstring of `PluginError` becomes "A plugin
could not be found, loaded, configured, started, or finish with a usable
result."

```python
class AtomicWriteError(OSError):
    kept: Path | None

    def __init__(self, cause: OSError, kept: Path | None = None) -> None: ...


def write_atomically(path: Path, data: bytes, *, make_parents: bool = True) -> None: ...
def retry_on_permission_error(action: Callable[[], T]) -> T: ...
def new_token() -> str: ...
def relative_to_run(run_dir: Path, path: Path) -> str: ...
def resolve_in_run(run_dir: Path, relative: str) -> Path: ...
```

- `write_atomically` is `document.py`'s private helper moved here
  (FM-38, FM-24: `O_EXCL`, mode `0o666`, `fsync`, `os.replace`
  retried 5 times 0.05 s apart on `PermissionError`), creating the parent
  folders first **unless `make_parents=False`**: the run document passes
  `False`, because a mistyped `--run-doc` folder is refused, not created
  (`test_a_folder_that_does_not_exist_is_a_run_document_error`, measured:
  without the option that test fails). It raises `AtomicWriteError`, which
  copies `errno`, `strerror` and `filename` of its `cause` (raised `from`
  it) and sets `kept` when only the rename failed. `retry_on_permission_error`
  is `document.py`'s `_retrying` moved with the constants it uses, since the
  document's read needs it too. `document.py` calls both and turns the error
  into its `RunDocumentError` with the same two messages as before. Its
  tests pass with one mechanical change: they patch `os.replace`, `os.fsync`
  and `time.sleep` through the module that imports them (`document_mod.os`),
  which is now `smc.run.files` (`files_mod.os`); every assertion stays.
- `new_token()` is `uuid.uuid4().hex[:12]`.
- `relative_to_run` gives `path` relative to `run_dir` as POSIX text and
  raises `ValueError` when it is not inside. `resolve_in_run` refuses an
  absolute text, a drive, a `..` part, and a result outside `run_dir`
  after `resolve()`, each with a `ValueError` naming the text (a run
  document copied from elsewhere must not make a stage read or write
  outside its folder).

### 3.7 Tests (#199)

`tests/unit/test_slots.py`: every model round-trips through
`model_dump(mode="json")` and `model_validate`, refuses `nan`, refuses an
unknown key, and `PlateResult` refuses `plate="glass"`;
`test_a_subclass_of_a_slot_model_is_refused_through_the_base` pins §2.3.
`tests/unit/test_geometry.py`: `pixel_matrix` and `stage_of_pixel` against
`PlateSample.render` on a non-square frame (FM-73) at 0°, 30° and 30°
mirrored, a blob's centroid mapping back within **0.05 px** (measured worst
0.001 px; the half-pixel centre of `nikon-control`, `W / 2`, is 0.708 px
off and swapped height and width 67.9 px, so "within a pixel" would not
fail the first);
`test_pixel_of_stage_inverts_stage_of_pixel`; `footprint` corners;
`well_geometry` against `WellPlatePlan` positions on a rotated plate, round
and square wells, with a margin. `tests/unit/test_run_files.py`:
`resolve_in_run` refusals (`..`, absolute, a drive on Windows given as
text), and `write_atomically`'s kept file on a failed rename.

## 4. Camera calibration (#22)

`smc/plugins/camera_calibration/algorithm.py` and `plugin.py`; entry point
`camera-calibration = "smc.plugins.camera_calibration.plugin:CameraCalibrationPlugin"`.
A port of `camera_cal.py` with its tests, plus a return check.

### 4.1 Algorithm

```python
UPSAMPLE = 10
MIN_RESPONSE = 0.05
DEFAULT_STEP_FRACTION = 0.25
FALLBACK_STEP_UM = 50.0


@dataclass(frozen=True, slots=True)
class Shift:
    d_col_px: float
    d_row_px: float
    response: float


@dataclass(frozen=True, slots=True)
class Fit:
    matrix_um_per_px: Matrix2
    pixel_size_um: float
    rotation_deg: float
    mirrored: bool
    residual_um: float
    notes: tuple[str, ...]

    @property
    def trustworthy(self) -> bool: ...


def measure_shift(reference: npt.NDArray[Any], moved: npt.NDArray[Any]) -> Shift: ...
def fit(
    shift_x: Shift, shift_y: Shift, *, step_um: float, image_shape_px: tuple[int, int]
) -> Fit: ...
```

- `measure_shift` returns the **displacement of the content** from
  `reference` to `moved`: content rolled by +7 rows and −13 columns gives
  `d_row_px=7`, `d_col_px=-13`. It uses
  `skimage.registration.phase_cross_correlation(reference, moved,
  upsample_factor=UPSAMPLE)` on frames cast to `float64`, mean-subtracted
  and divided by their standard deviation, and negates what it returns
  (skimage gives the shift that registers `moved` onto `reference`).
  `response` is the absolute Pearson correlation of the two frames'
  overlap **after** integer alignment (the regression of
  `test_confidence_is_measured_after_alignment`), 0 when the overlap is
  under 8 pixels on an axis or either frame has no variance. A frame that
  is not 2-D, or two frames of different shapes, is a `ValueError` (the
  layer's `Camera.snap` is 2-D).
- **A shift beyond half the frame wraps.** skimage reports each axis's
  shift in `[−N/2, N/2)`: content moved by 200 rows on a 384-row frame
  reads −184 (measured, scikit-image 0.26). The response, measured at the
  wrapped shift, is then that of unrelated content (0.005, measured), so
  the fit refuses it instead of returning a wrong matrix; the plugin keeps
  its default step well inside the frame (§4.2) and says so when a step may
  have wrapped.
- `phase_cross_correlation` is not annotated in scikit-image 0.26, and
  `mypy --strict` refuses the call (`no-untyped-call`). The module binds it
  once, `_phase_cross_correlation: Callable[..., tuple[Any, Any, Any]] =
  registration.phase_cross_correlation`, which mypy accepts (measured); no
  `type: ignore`, no override in `pyproject.toml`. Importing
  `skimage.registration` costs 0.11 s (measured), so it is imported at the
  top of the module.
- `Fit.trustworthy` is `not notes`: the issue's
  `CameraCalibration.trustworthy`. A `CameraCalibration` in a `done` slot
  is trustworthy by construction (§2.3), so the property lives on the fit.
- `fit`: `S` has the columns `(d_col, d_row)` of the +X and the +Y probe,
  `J = S / step_um`, and `M = −inv(J)` (§2.1). `pixel_size_um =
  sqrt(|det M|)`, `mirrored = det M > 0`, and `rotation_deg =
  atan2(A[1][0], A[0][0])` with `A = M · diag(1, −1)`, its first column
  negated first when mirrored, in `(−180, 180]`. `residual_um` is the
  largest entry of `|M − ideal|` times `max(H, W) / 2`, where `ideal` is
  the scaled rotation (mirrored if so) of that size and angle: how far the
  frame's edge is from a rigid camera. Notes, each making the result
  unusable: `min(response) < MIN_RESPONSE` ("the frames barely correlate:
  find a field with texture"); `|det J| < 1e-12` ("the two moves gave
  parallel shifts: did the stage move?"), with a zero matrix and no
  division; and `max|M − ideal| >= pixel_size_um` ("the shifts do not fit
  a rigid camera: did each axis move its full command?"), which refuses
  an axis that moves a fraction of its command, or none plus noise
  (review of #211).
- **A frame with no signal** (#408, F8). With the light path on the
  eyepiece the camera sees its black level and its fixed pattern, which
  correlate well at a zero shift: the two probes measured a shift of zero,
  the response was high, and the only note was "parallel shifts: did the
  stage move?". `signal_spread(frame)` is the 99th minus the 1st
  percentile of the frame in counts (float64, so a frame that is not 2-D
  or not finite is a `ValueError`), and `carries_signal(frame,
  bit_depth)`, the **one** decision every caller asks (the reference and
  each probe frame), is true when that spread is `>=
  MIN_SIGNAL_SPREAD_COUNTS`, **50** at 16 bits (`signal_floor_counts`
  scales it by `2 ** (bits - 16)`, so 3.125 at 12 bits, and never above
  16: a shallower mode reads fewer counts for the same light, and the
  scaling can only let a dark frame through, the old behaviour, never
  refuse a lit one), **or** when at least `MIN_FEATURE_FRACTION`
  (`1e-4`, 20 pixels of a 384 x 512 frame, 199 of 1412 x 1412) of the
  pixels lie a floor or more from the median. The second rule is for a
  **sparse** field, a few beads or cells on a dark background: three
  beads of 2.5 px are 0.24 % of the frame, and the percentiles read it as
  dark (a spread of 10 or 11), where the base code calibrates it (found
  by the adversarial review). The two real dark frames have no such
  pixel (0 of 2 million), and a hot pixel is one in a million. Measured on the Prime 95B (16-bit):
  the two dark frames of Ti2-E session 2 (`smc snap`, 12 and 20 ms) span
  **11** counts (mean 100.5, std 2.3, `max - min` 45 to 47); of the 200
  frames of the stand's survey, the only ones under 50 are those two and
  two saturated constant frames (0), and the dimmest lit one spans
  **177**; the 4x scan fields span 395 to 1123 and B3 at the 10x 5319. It
  is the percentiles because one hot or stuck pixel makes a dark frame's
  `max - min` 65000; it is counts because a ratio to the pixel noise
  cannot tell a dark frame from a lit blank one (5.2 and 5.7, near the
  4.65 of Gaussian noise). The floor was measured on a 16-bit camera, the
  only kind on our stands, and follows the bit depth the camera reports.
  A frame at full scale (its median is `2 ** bits - 1`) is refused with
  its own remedy, "the camera's frame is saturated (...): lower the
  exposure or the light" (one message, many causes). A lit field without texture still passes (the synthetic plate
  with `texture_std=0` spans 94): it is refused later, by the fit, as "the
  frames barely correlate".
- **A probe that fits poorly** (#408, F6). `precision_reasons(fit,
  step_um, image_shape_px)` names two reasons, never refusing: the step at
  the **measured** pixel size is under `SMALL_STEP_FRACTION` (a tenth) of
  the smaller side (`18 px, under a tenth of the 1412 px frame`; 141 px on
  the Prime), and the residual is over `MAX_RESIDUAL_SHARE` (5 %) of the
  step (`residual 10.7 µm, 21 % of the step`). Session 3's four fits: 50 µm
  at the 4x with no pixel size known, 18.2 px and 21 %; 900 µm at the 4x,
  329 px and 0.1 %; 350 µm at the 10x, 311 px and 0.2 %; 100 µm at the
  40x, 362 px and 0.5 %. The synthetic plate does not reproduce the 4x's
  residual (0.000 at 18 px), so the residual rule is tested on these
  numbers and the pixel rule through the plugin.

### 4.2 Plugin

```python
class CameraCalibrationParams(PluginParams):
    step_um: float | None = Field(None, gt=0.0, le=2000.0)
    settle_ms: float = Field(0.0, ge=0.0, le=10_000.0)
    from_run_doc: Path | None = None
```

`requires = (XYStage, Camera)`, `reads = reads_optional = ()`,
`provides = "camera-calibration"`. The objective is
`microscope.get(ObjectiveTurret).current()`, `None` without a turret.

**Reuse.** With `from_run_doc`, nothing moves, and a dry run may reuse:
the other document is loaded (`RunDocument.load`, its errors become
`PluginRunError`), and `other.unusable("camera-calibration",
profile=<this profile>, dry_run=ctx.dry_run)` must be `None`: the
runner's own rule and messages (not `done`, another profile, a dry run
read by a real one). Then the objective in place (when there is a turret)
and `camera.image_shape()` must equal the result's, else `PluginRunError`
naming both values. The result is rebuilt through
`CameraCalibration.model_validate` (never `model_copy`, which skips
validation) with `reused_from` the given path's `as_posix()` text, as
typed and not resolved (FM-50: an absolute path names a user folder),
`return_error_um=None`, and `assumed` gaining `camera calibration reused
from <path>, measured <finished_utc, ISO, seconds>` after the reused
result's own lines. A result that does not validate is a `PluginRunError`
that says to run `camera-calibration` again there.

**Measurement**, refused in a dry run (`PluginRunError`: "a dry run cannot
measure the camera; run it without --dry-run"):

1. The step: `step_um`, else `DEFAULT_STEP_FRACTION × min(height, width)
   × pixel size` when `Camera.pixel_size_um()` is finite and positive (the
   smaller side: a rotated camera puts the X probe on either image axis,
   and a shift beyond half a side wraps, §4.1), else `FALLBACK_STEP_UM`
   with `assumed` gaining `step of 50 µm: no pixel size was known` (a probe
   distance, not a result).
2. `xy.wait()` (FM-85: a stage still travelling from an earlier move
   reads a fraction of it); `start = xy.position_um()`; `reference =
   camera.snap()`, **judged before anything moves**: a shape other than
   the camera reports and a value that is not finite are refused with
   `nothing was moved`, and so is a frame with no signal (§4.1): `the
   camera's frame is dark or flat (the 1st to 99th percentile spans <s>
   counts, under 50): is the light on and the light path on the camera?
   nothing was moved` (`under` is the floor of the camera's bit depth; a
   frame at full scale reads `the camera's frame is saturated (...): lower
   the exposure or the light; nothing was moved`).
3. For `(dx, dy)` in `(step, 0)`, `(0, step)`: `ctx.raise_if_stopped()`,
   `xy.move_to_um(start + (dx, dy))`, `ctx.wait_s(settle_ms / 1000)`,
   `snap`, `measure_shift(reference, frame)` and `signal_spread(frame)`;
   `ctx.progress(i, 3, "+X
   probe" | "+Y probe")`. On any exception here (a stop, a halt, a fault,
   a soft-limit refusal) the stage is left where it is: a WARNING names
   where (`position_um()` read in its own `try`, "unknown" when that read
   fails too, so it never hides the first error) and the start, and the
   exception propagates (FM-88, FM-64).
4. Both probes returned normally, so the return runs (FM-88) **before the
   fit is judged**: `ctx.raise_if_stopped()`, `move_to_um(start)`,
   `ctx.wait_s(settle_ms / 1000)`. An operator who calibrated on an empty
   field gets the stage back where they chose it.
5. **A probe frame with no signal** ends the run, after the return and
   before the fit, so that "parallel shifts" follows only frames that carry
   signal: `the frame after the <+X probe|+Y probe> is dark or flat (the
   1st to 99th percentile spans <s> counts, under 50): did the light go
   off, or did the step of <s> µm leave the lit sample? the stage was sent
   back to its start` (the first such probe is named; a probe that left the
   lit sample, onto the black plastic between wells, reads the same). The
   return was sent and came back normally; where it landed is not read
   here (FM-167), so the text says what was done, not where the stage is.
   Then `fit(...)`; notes → `PluginRunError` whose reason is the notes, then
   `step <s> µm`, and, when the step was the fallback or is at least 40 %
   of the smaller side in pixels (`step / pixel size`, pixel size known),
   `a step over 40 % of the field may wrap: give a smaller step_um`.
   Nothing fills the slot. A trustworthy fit that `precision_reasons`
   names adds one line to `assumed`, after the fallback or cap line and
   before the return's: `step of <s> µm fits poorly (<reasons, "; ">): run
   again with step_um=<q>`, where `q` is `suggested_step_um`, a quarter of
   the smaller side at the measured pixel size to two significant figures
   and at most 2000 (970 for session 3's 4x), and the clause is left out
   when `q` is not larger than the step used. At most 160 characters, the
   length the page's strip keeps of the first line (FM-205).
6. `snap`; `measure_shift(reference, frame)`; `ctx.progress(3, 3,
   "return")`. When its response is at least `MIN_RESPONSE`,
   `return_error_um` is its stage length (`M` applied); otherwise `None`,
   and `assumed` gains `the return could not be checked: the frame after
   it did not correlate (<r>)`. A return error above `max(2.0, 4 ×
   pixel_size_um)` µm adds to `assumed` `the stage came back <n> µm off:
   its repeatability limits the scan's stitching` (reported, not refused).
7. `response` is the smaller of the two probes' responses. When
   `Camera.pixel_size_um()` is finite and positive and differs from the
   measured pixel size by more than 2 %, a WARNING says `the camera reports
   <r> µm per pixel for <objective>, measured <m>: update the profile's
   pixel size`.

**The issue's "`Camera.pixel_size_um` can be set from it per objective"**
is met without writing the layer: the `Camera` protocol has no setter, and
a stage that changed the stand's configuration as a side effect would be
hidden state. The scan plan takes the matrix from this slot when its
objective and shape match (§7.3), a calibration per objective is one run
per objective (or a reuse), and the step-7 WARNING tells the owner which
profile value to change.

### 4.3 Tests

Ported, `tests/unit/test_camera_calibration.py`:
`test_a_known_shift_is_recovered`, `test_confidence_is_measured_after_alignment`,
`test_a_blank_field_scores_zero`, `test_pixel_size_and_handedness_are_recovered`
(the synthetic convention: upright is not mirrored),
`test_rotation_is_recovered`, `test_a_stage_that_does_not_move_is_caught`;
new: `test_a_shift_beyond_half_the_frame_is_refused_not_returned`.
`tests/test_camera_calibration_demo.py`, on `demo_microscope_with_sample`
with a textured `PlateSample`:
`test_calibration_recovers_a_rotated_mirrored_camera` (a 384 × 512 frame,
30°, mirrored, pixel size within 2 %, rotation within 0.5°);
`test_a_featureless_field_fails_fills_no_slot_and_returns_the_stage`;
`test_a_stop_leaves_the_stage_where_it_is`,
`test_a_halt_leaves_the_stage_where_it_is`,
`test_a_fault_leaves_the_stage_where_it_is` (FM-88's three);
`test_a_dry_run_refuses_to_measure`;
`test_reuse_copies_a_done_calibration_and_says_so` (also in a dry run);
`test_reuse_refuses_another_profile_objective_or_shape`;
`test_a_misreported_pixel_size_is_logged`; and for #408
`test_a_frame_with_no_signal_is_refused_before_any_move_and_the_light_path_is_named`,
`test_a_probe_frame_with_no_signal_is_refused_after_the_return_not_blamed_on_the_stage`,
`test_a_fallback_step_of_a_few_pixels_is_named_with_a_suggested_step`.

**Failure modes**: FM-88, FM-64, FM-73, FM-75 (casts), FM-104 (finite
pixel size), FM-85 (wait before the start is read; readbacks after the
facade's move only), FM-50 (`reused_from` as typed).

## 5. Plate registration and going to a well (#19)

`smc/plugins/plate_registration/algorithm.py` and `plugin.py`
(`plate-registration`), and `smc/plugins/goto_well.py` (`goto-well`). A
port of `plate.py` with its 31 tests.

### 5.1 Guided steps without hidden state (decides §1.13)

The references are **parameters**, not state. The guided loop of
`docs/scope-bringup.md` §7 becomes:

1. Put A1 roughly under the objective; `smc run plate-registration
   --param plate=96-well --param 'refs=[{"well": "A1", "x_um": ..., "y_um":
   ...}]'`, with the position from `smc stage get`. One reference is a
   **seed**: `method="seed"`, rotation 0, and it fills `plate`.
2. `smc run goto-well --param well=H12` drives there through the latest
   `plate`; centre the well by eye (jog), read `smc stage get`.
3. Run `plate-registration` again with every reference so far. Each run
   replaces `plate`; the run document's `params` keep every reference that
   made it, which is the audit trail `nikon-control`'s `plate.json` lacked.

Wall touches go the same way: `goto-well --param side=-X`, then a
`WallTouch` per wall in `walls` (the sides are stage directions, §19). The plate page (§15) collects the
references by clicks and calls the same plugins with the same parameters.
Nothing survives between two runs that the document does not show.

### 5.2 Algorithm

```python
SCALE_TOLERANCE = 0.02
GOOD_RESIDUAL_UM = 300.0
DIAMETER_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class EdgeCentre:
    well: str
    x_um: float
    y_um: float
    measured_width_um: float | None
    measured_height_um: float | None
    assumed_axes: tuple[str, ...]


def calibrate(plate: str, refs: Sequence[WellRef]) -> PlateResult: ...
def centre_from_walls(
    plate: str, well: str, touches: Sequence[WallTouch]
) -> EdgeCentre: ...
def nominal_offsets(plate: str) -> dict[str, tuple[float, float]]: ...
def wall_position(result: PlateResult, well: str, side: str) -> tuple[float, float]: ...
def nearest_well(
    result: PlateResult, x_um: float, y_um: float
) -> tuple[str, float]: ...
def suggested_refs(plate: str) -> tuple[str, ...]: ...
```

As `plate.py`: a rigid fit (Kabsch) with the scale fixed by the plate
definition; one reference sets A1 with rotation 0; the scale check
compares the longest baseline with the nominal and raises `ValueError`
beyond `SCALE_TOLERANCE` ("off the 96-well definition by 3.1 %: wrong
plate type, or a misidentified well"); a duplicate well (after
`normalise_well`), an unknown well and no reference raise `ValueError`.
`centre_from_walls` takes the midpoint of an opposite pair (and measures
the size) or one wall plus the nominal radius, needs one X and one Y wall,
and raises `ValueError` when an implied size is off by more than
`DIAMETER_TOLERANCE`. `nearest_well` returns the name and the distance
from its centre.

### 5.3 Plugins

```python
class PlateRegistrationParams(PluginParams):
    plate: str = Field("96-well", description="A useq plate name.")
    refs: tuple[WellRef, ...] = ()
    walls: tuple[WallTouch, ...] = ()
```

`requires = ()`, `reads = reads_optional = ()`, `provides = "plate"`.
`run` turns each well's walls into a reference (`centre_from_walls`), then
calls `calibrate`. Its `ValueError`s, no reference at all, a well given
both as a reference and by walls, and a `residual_um` above
`GOOD_RESIDUAL_UM` ("<well> is <n> µm from where the others put it:
centre it again") become `PluginRunError`. An assumed axis of a wall
centre goes into `assumed`.

```python
class GotoWellParams(PluginParams):
    well: str
    side: Literal["centre", "+X", "-X", "+Y", "-Y"] = "centre"  # §19
```

```python
class GotoWellResult(PluginResult):
    well: str
    side: str
    target_x_um: float
    target_y_um: float
    x_um: float
    y_um: float
```

`goto-well`: `requires = (XYStage,)`, `reads = ("plate",)`, `provides =
"goto-well"`; one `move_to_um` to the well's centre or wall point, whose
soft limits and dry run are the layer's. Its model lives in
`goto_well.py`: nothing reads its slot.

### 5.4 Tests

The 31 ported tests (`test_plate_registration.py`), with
`test_calibration_round_trips_through_useqs_forward_model` as a hypothesis
test (plate type, A1, rotation within ±8°, 1 to 4 references with ±12 µm
of jitter; the bounds are in the block below);
`test_a_seed_fills_the_plate_slot_with_rotation_zero`;
`test_a_residual_beyond_300_um_fails_and_names_the_well`;
`test_a_well_given_twice_is_refused`; `test_goto_well_drives_to_the_wall_point_on_a_rotated_plate`
(demo); `test_goto_well_needs_a_plate`. `docs/plugins/plate-registration.md`
documents §5.1's loop.

**Failure modes**: FM-73 (anisotropic probes, a rotated plate),
FM-43, FM-121 (the rotation bound below), FM-124.

**Refined while planning #19** (measured 2026-10-28 on a prototype that
is not committed, useq-schema 0.9.2 and pydantic 2.13.5 on Python 3.13;
`useq._plate_registry` read in the 0.9.0 wheel too):

- **Two references can fit a plate turned by 90° with no residual.** A1 at
  (0, 0) and A2 at (9000, 0) µm, with A2 typed as `B1`: the fit gives
  90.0°, a residual of 3e-13 µm and a scale error of 0, and puts H12 at
  (63 000, 99 000) µm instead of (99 000, −63 000). Neither the scale check
  nor the residual can see it. `calibrate` therefore takes
  `max_rotation_deg: float = MAX_ROTATION_DEG` (keyword, `MAX_ROTATION_DEG =
  10.0`) and raises `ValueError` beyond it: `the fit turns the plate by
  <r>°, beyond ±<max>°: a reference well is misidentified (two references
  the same distance apart fit either way)`. `PlateRegistrationParams` gains
  `max_rotation_deg: float = Field(10.0, gt=0.0, le=45.0)`, for a holder
  that really holds a plate askew (FM-121: the bound is on the derived
  value, not only on the inputs).
- **Nominal offsets come from useq.** `nominal_offsets(plate)` is the
  `all_well_positions` of a plan with A1 at (0, 0) and rotation 0, keyed by
  `str(name)`: 3 ms for the 1536 wells, where one `well_geometry` per well
  takes 0.34 s. `nearest_well` uses `result.to_plan().all_well_positions`
  the same way. A fit of 3 references with no jitter gives A1 back within
  2e-11 µm and the rotation within 4e-15° on the 6-, 24-, 96-, 384- and
  1536-well plates at 7.25°.
- **The fit** (`plate.py`'s): centroids of the nominal offsets `d` and the
  measured points `m`, `θ = atan2(Σ d′ × m′, Σ d′ · m′)` of the centred sets,
  `A1 = m̄ − R(θ) · d̄`. The scale check (the longest nominal baseline among
  the references, measured over nominal minus 1, signed) runs before the
  fit and the rotation bound after it. The residual is computed from the
  fitted A1 and θ before the `PlateResult` is built: the result is
  constructed once, never `model_copy`'d (which skips validation).
- **A seed** (one reference) has `residual_um=None`, `scale_error=None`
  (§3.2: nothing measured them; `plate.py` wrote 0.0) and `assumed` gains
  `one reference well (<well>): rotation 0° assumed, not measured`.
- **The residual spreads.** One reference of three that is 600 µm off
  gives a residual of 298 µm; 700 µm gives 348 µm (A1, A12 and H1 of a
  96-well plate, H1 moved in X). The worst reference is the one named.
- **Wall touches are read per axis**, as `plate.py`: `-X` and `+X`
  (`left` and `right` before #327, §19) give X from their `x_um`, `-Y` and
  `+Y` (`bottom` and `top`) give Y from their `y_um`,
  and the other coordinate of a touch is not used, since the plate's
  rotation is not known yet when walls are touched. `+Y` is towards
  row A: rows grow towards −Y. An opposite pair gives the midpoint and a
  signed size (`+X − (−X)`, `+Y − (−Y)`, so swapped sides are
  refused by the size check); one wall gives the wall plus or minus the
  nominal half-size of that axis (`well_size[0]` for X, `[1]` for Y), and
  its axis is assumed. Touching along a stage axis from the wall point of
  a plate turned by θ errs by `r (1 − cos θ)` for one wall: 8 µm at 4°,
  49 µm at 10° for a 96-well well (computed); a pair cancels it for a round
  well. A side given twice and a touch of another well are `ValueError`s.
- **`wall_position`** is the well's centre plus `R(θ) · (∓w/2, 0)` for
  `-X` and `+X` and `R(θ) · (0, ∓h/2)` for `-Y` and `+Y`, `w` and `h`
  the well's size; any other side is a `ValueError` (`plate.py` raised a
  bare `KeyError`).
- **`refs` of the result** hold the given references, then one per well
  given by walls (its computed centre), in the order of the well's first
  touch, all normalised.
- **`goto-well` in a dry run** returns the target from `move_to_um`
  (measured: the demo's dry run returns the target and the readback stays
  where it was), so `x_um`, `y_um` are the target and `assumed` gains `dry
  run: the stage did not move; x_um and y_um are the target`. The demo's
  stage lands within 0.01 µm of a fractional target (measured:
  18123.4567 → 18123.45). A `plate` from a dry run is unusable to a real
  `goto-well` (§1.7's rule), which the plugin page says.
- **The hypothesis test's bounds** are proved, not fitted: the rigid fit
  minimises the sum of squares, so with jitter of at most `J` on `n`
  references the residual is at most `√n · J`, and each reference's true
  centre is predicted within `(√n + 1) · J` (5000 random cases: 1.39 J and
  1.27 J at worst). The rotation is drawn within ±8°: with it drawn
  within ±10°, jitter on close references gave a fit of −10.0° that the
  bound refused (measured).
- **An anisotropic plate for FM-73** is registered by the test only,
  `monkeypatch.setitem(useq._plate_registry._PLATE_REGISTRY, name, {...})`
  (the module and the dict exist in 0.9.0 and 0.9.2): every registered
  plate has equal pitches and well sizes on both axes, so an exchange of X
  and Y passes on all of them.
- **On the Ti2-E** the profile's soft limits are ±49 000 × ±29 000 µm, a
  span of 98 × 58 mm, and a 96-well plate's corner wells are 99 × 63 mm
  apart: wherever the plate sits, a `goto-well` to some corner wells will
  be refused by the layer (not measured at the stand; #198).

## 6. Automatic plate finding (#20)

`smc/plugins/plate_finding/algorithm.py` and `plugin.py`, entry point
`plate-finding`. A port of `plate_find.py` with its tests; it fills
`plate` with `method="lattice-fit"` and `fit` set.

### 6.1 Algorithm

The four-stage search of `plate_find.py`, unchanged, with its constants
(`INSIDE_FRACTION`, `MAX_ROTATION_DEG = 4.0`, `DECISIVE_MARGIN = 0.08`,
`MIN_SCORE = 0.25`, `ROTATION_CANDIDATES = 3`, `DEFAULT_JITTER = 0.5`,
`STEP_FRACTION = 1/3`, margin 1 pitch):

```python
@dataclass(frozen=True, slots=True)
class SurveyPoint:
    x_um: float
    y_um: float
    value: float


def survey_points(
    plate: str,
    centre_um: tuple[float, float],
    *,
    step_um: float | None = None,
    margin_pitches: float = 1.0,
    jitter: float = DEFAULT_JITTER,
    seed: int = 0,
) -> list[tuple[float, float]]: ...
def suggested_step_um(plate: str) -> float: ...
def fit(
    points: Sequence[SurveyPoint], plates: Sequence[str] | None = None
) -> list[PlateFit]: ...
def decide(fits: Sequence[PlateFit]) -> tuple[PlateFit | None, tuple[str, ...]]: ...
```

`PlateFit` keeps `plate_find.py`'s fields; `PlateFit.to_result(*,
runner_up, margin) -> PlateResult` gives `residual_um=None`,
`scale_error=None`, `n_refs=0` (not `nikon-control`'s `max(2,
n_inside)`, which claimed references nobody took) and the `LatticeFit`.

### 6.2 Plugin

```python
class PlateFindingParams(PluginParams):
    plate: str = "96-well"
    plates: tuple[str, ...] = ()
    centre_x_um: float | None = None
    centre_y_um: float | None = None
    step_um: float | None = Field(None, gt=0.0, le=20_000.0)
    margin_pitches: float = Field(1.0, ge=0.0, le=5.0)
    jitter: float = Field(0.5, ge=0.0, le=1.0)
    seed: int = Field(0, ge=0)
    statistic: Literal["mean", "std"] = "mean"
    max_failed_in_row: int = Field(3, ge=1, le=100)
```

`plate` sets the survey's extent; `plates` the candidates fitted (empty:
every registered `-well` plate). The centre is both coordinates or
neither (a model validator), each within ±1 000 000 µm (FM-32); neither
is the stage's position. `max_failed_in_row` is `scan`'s (§8), and is
decided in *Refined while planning #248* below.

- **The autofocus.** It cannot lock over the plastic between wells. When
  it is engaged, the plugin calls `disengage()` (a safe call, no motion)
  before the first move, logs a WARNING, and never engages it again
  (FM-88: engaging drives Z). What the result says about it is the
  autofocus's state at the end of the survey, decided in *Refined while
  planning #248* below.
- **The loop**: per point `ctx.raise_if_stopped()`, `move_to_um`, `snap`,
  the statistic, `ctx.progress`. A `SafetyRefusedError` (a point outside
  the soft limits) and any other `HardwareError` except the halt and the
  motion stop skip the point, counted; more than 10 % skipped, or
  `max_failed_in_row` in a row, is a `PluginRunError`. The survey is the
  checkpoint (`SurveyCheckpoint`, below), saved before the first move,
  every 50 points and at the end, so a failed fit keeps it and `--resume`
  refits without moving (§1.4).
- `decide` returning `None` is a `PluginRunError` with its notes; a
  "barely beats" note goes into `fit.notes` of a usable result. A dry run
  is refused as in §4.

**Tests**: the ported `test_plate_finding.py` (synthetic surveys); new
`test_the_survey_fills_the_plate_slot_on_the_demo` with a `PlateSample`
and a coarse step; `test_an_engaged_autofocus_is_switched_off_and_not_relocked`
(fake); `test_a_failed_fit_keeps_the_survey_for_a_resume`. **Failure
modes**: FM-64, FM-88, FM-12, FM-141, FM-142.

**Refined while planning #20** (measured 2026-10-28 on a prototype that is
not committed: `plate_find.py` copied from `nikon-control@1aad7c9`, which
holds it unchanged, with useq-schema 0.9.2, numpy 2.5.3, pydantic 2.13.5,
pymmcore-plus 0.18.1, Python 3.13):

- **The original tests pass on the copy**, all 16 that do not need
  `nikon-control`'s `plate` and `wells`, also with `-W error::RuntimeWarning`.
  They take 17 s together. One `fit` over the seven `-well` plates takes
  about 2 s on a 96-well survey of 1120 points, and 7.9 s on a 1536-well
  survey of 14 800 points.
- **Units in names.** `PlateFit` has the fields `plate`, `a1_center_xy_um`,
  `rotation_deg`, `score`, `polarity: Literal[1, -1]`, `n_samples`,
  `n_inside` and `notes`, and keeps `usable` and `describe()`. The helpers
  the tests use are public: `plate_pitch_um`, `plate_shape` (rows, columns),
  `well_size_um`, `survey_bounds_um`, `grid_points`, `inside_mask`,
  `lattice_score` and `fit_one`. The search's numbers (spans, step counts,
  the order of the stages) do not change. The geometry may be computed once
  per `fit_one` instead of once per score (about 0.5 s of the 2.9 s is
  useq's validation), if the results stay the same.
- **The survey is walked in snake order.** `grid_points` makes the same
  points from the same seed as `plate_find.py`, row by row from the lowest Y,
  and then reverses every other row. The demo stage moves at its `Velocity`
  property, 10 mm/s by default: 3 mm takes 0.30 s and 30 mm 3.0 s. The
  original raster ends every row with a return across the whole survey, 27
  returns of 117 mm on a 96-well plate. The fit does not depend on the order.
- **A bound on the number of points**, `MAX_SURVEY_POINTS = 25_000` (FM-74,
  FM-32). `survey_points` counts the points in Python integers before it
  makes any, and raises `ValueError` beyond the bound: `the survey would take
  <n> points, more than 25000: give a larger step_um or a smaller
  margin_pitches`. At the default step and margin a plate takes 130 (6-well),
  208, 352, 616, 1120, 3952 and 14 800 (1536-well) points. At a margin of 5
  the 1536-well plate takes 21 328.
- **The extent check is a function of its own.** `survey_extent_note(points_um:
  Sequence[tuple[float, float]], plate: str) -> str | None` is
  `plate_find.py`'s check, and returns its text. `fit_one` puts that text in
  its notes, as before. The plugin calls it twice, and each time a note is a
  `PluginRunError`:
  1. **Before anything moves**, on the planned points inside the soft limits,
     for `params.plate`.
  2. **After the fit**, on the sampled points, for the winner's plate. A
     skipped edge, or a winner larger than the plate surveyed, can leave A1
     unpinned. A fit that may be whole wells off is not usable.

  A survey without margin still placed a 96-well plate within 440 µm while
  the check reported it. Whether A1 is right is exactly what such a survey
  cannot show, so it is refused, not reported.
- **Soft limits are applied before the survey.** When `XYStage.limits_um()`
  is not `None`, a planned point is kept when `low ≤ x ≤ high` on both axes,
  which is the layer's own test (`check_xy_target_um`). The rest are not
  visited, and `assumed` gains `<n> of <N> survey points are outside the
  stage's soft limits and were not sampled`. On the demo and the fake,
  `limits_um()` is `None`.
- **On the Ti2-E as profiled, every plate is refused before it moves.** The
  soft limits span 98 × 58 mm. The extent check needs the wells' span plus
  one pitch: 108 × 72 mm for the 96- and 384-well plates, 114 × 76 mm for the
  24-well plate (6 × 4 wells at a 19 mm pitch). Whether the limits can be
  widened is the owner's call at the stand (#198). At one pitch of margin, a
  96-well survey (117 × 81 mm, ± 1.5 mm of jitter) stays inside the plate's
  SBS footprint of 127.8 × 85.5 mm. A 6-well survey (156 × 117 mm) and any
  larger margin do not.
- **The checkpoint** is `SurveyCheckpoint(centre_um: tuple[float, float],
  total: int, next_index: int, skipped: int, points: tuple[SurveyPoint,
  ...])` (`total` is decided in *Refined while planning #248* below).
  `next_index` counts the planned points handled, sampled or skipped, in the
  list left after the soft limits. The centre is kept because without
  parameters it is the stage's position at the start, and on a resume the
  stage is elsewhere. **It is saved before the first move** (FM-141), then
  every 50 points and at the end: a survey stopped before its 50th point
  would otherwise leave no centre, and the resume would survey around
  wherever the stage stopped (reproduced in the review of #20: 208 points
  from −112 mm to +21 mm in x, `done`, nothing in `assumed`). A resume
  rebuilds the planned list from the parameters and the kept centre. It is
  refused when `next_index` exceeds the list's length, or `skipped +
  len(points) != next_index`, with `run plate-finding without --resume`. A
  pydantic model built from Python objects does not validate a nested
  dataclass instance again: a `SurveyPoint` holding `nan` is accepted, and
  refused only when the JSON is read back. So the loop never builds a
  `SurveyPoint` whose value is not finite. Such a point is a skipped point
  (`the frame's <statistic> is not finite`).
- **A point is where the stage says it was** (FM-142). `move_to_um` returns
  the position reached, and the `SurveyPoint` is built from it, not from the
  target. A stage that stops short of a target without an error (clamped at
  45 mm in the test) gave a survey of the 12-well plate that was `done`,
  scored 0.6, had A1 26 mm (a pitch) off and an empty `assumed`; placed
  where the stage says it was, the survey spans 90 mm and the extent check
  refuses it. A readback that is not finite is a skipped point (`the
  stage's readback is not a finite position`).
- **Bounds the plan left out** (FM-32, from the review of #20). `seed >= 0`;
  a centre coordinate within ±1 000 000 µm (no stage travels a metre: a
  centre that far out is a unit slip); a plate named twice in `plates` is
  refused (`plate-finding fits each plate once`). A survey with fewer than
  2 points inside the soft limits is refused before moving (`only <n> of
  <N> survey points are inside the stage's soft limits`). Limits that cut
  nothing say nothing, and a point exactly on a limit is visited.
- **Skips.** `skipped * 10 > total`, checked at each skip in integers, where
  `total` is the length of the planned list, saves the checkpoint and raises
  `PluginRunError`: `<k> of <N> survey points could not be sampled, the last
  at (<x>, <y>) µm: <Type>: <first line>; check the stage, then run
  plate-finding --resume`. When the survey ends, `assumed` gains `<k> of <N>
  survey points could not be sampled and were skipped` for `k > 0`.
- **The statistic** is `frame_value(frame, statistic) -> float` in the
  algorithm: `np.mean` or `np.std` with `dtype=np.float64`.
- **The autofocus line** is `the autofocus was switched off for the survey;
  engage it again before the scan`, logged as a WARNING when it is switched
  off (since #248 the result's own line is the measured one, below). The
  autofocus is switched off only when `is_engaged()` and at least one point
  is left to visit: a resume that only fits moves nothing. On the fake,
  `disengage()` logs `enableContinuousFocus(False)`.
- **The result.** `PlateFit.to_result(*, runner_up: str | None, margin:
  float | None, notes: Sequence[str], assumed: Sequence[str] = ()) ->
  PlateResult`. `notes` are `decide`'s, which include the winner's own notes
  and the "barely beats" note. The result is built once, never
  `model_copy`'d. `runner_up` and `margin` (the winner's score minus the
  runner-up's) are `None` when one plate was fitted.
- **No move after the survey.** The stage stays at the last point. Nothing
  returns, and the autofocus is never engaged again (FM-88).
- **The demo test** sets the demo stage's `Velocity` to `100000` through
  `microscope.core`. A 12-well `PlateSample` with A1 at (−40 000, 25 000) µm,
  rotated by 1.2°, imaged as 24 × 32 frames at 20 µm per pixel, and surveyed
  from a centre (700, −500) µm off: 208 points in 3.0 s, a fit in 1.5 s, the
  12-well plate identified (score 0.88, runner-up 0.60), and the worst well
  centre 0.23 of a well radius off. At the default velocity the same survey
  takes 350 s. A 96-well survey takes 16 s even at the high velocity (14 ms
  per point), too long for the default suite.

**Refined while planning #248** (2026-10-29; each behaviour below was
reproduced on the merged code with a throwaway test on the fake core, and
the pass rates of the margin on `survey_points` and `survey_extent_note`
over eight seeds):

- **A stage that fails every move ends the survey at the third.** Today
  three failed moves in a row leave a `done` survey with 205 samples, and a
  stage that fails every move ends it only at the 21st of 208 points, when
  `skipped * 10 > total`. On a 96-well survey that is 113 failures, 1.9 h at
  the 60 s move timeout of the shipped profiles. `max_failed_in_row` (default
  3, 1 to 100, as `scan`'s, §8) counts the skipped points in a row of every
  kind: a failed move, a frame whose value is not finite, a readback that is
  not finite. A sampled point resets it, and a resume restarts it at 0 (the
  operator checked the stage). It is checked first at each skip; either
  rule saves the checkpoint (`next_index = i + 1`) and raises
  `PluginRunError`: `<n> survey points in a row could not be sampled, the
  last at (<x>, <y>) µm: <Type>: <first line>; check the stage, then run
  plate-finding --resume`. The text of the rule of a tenth does not change.
- **The checkpoint keeps the survey's size.** `SurveyCheckpoint.total`
  (`int`, `ge=0`, required) is `len(kept)`, the points left inside the soft
  limits. The checkpoint's `next_index` indexes the list `kept`, which is
  rebuilt from the limits each attempt reads, and the runner compares the
  profile's name, not its `[safety]`. Reproduced: stopped at 50 of 154
  points on limits that kept 154, resumed on limits that keep 173: `done`,
  173 samples, every index shifted. A resume is refused when `total`
  differs from this attempt's, before the other two checks: `the checkpoint
  was made for a survey of <a> points inside the soft limits and this one
  has <b>: the stage's soft limits changed; run plate-finding without
  --resume`. A checkpoint without `total` (made before #248) is refused by
  `ctx.checkpoint` as any the model does not accept. The count is not a
  fingerprint: limits that move and keep the count pass it, which takes two
  edits at once and is not guarded.
- **What the result says about the autofocus is measured at the end.** A
  survey that switched the autofocus off, was stopped, and was resumed ended
  `done` with an empty `assumed` (reproduced): the line died with the first
  attempt, and the resume found the autofocus off. Recording the switch-off
  in the checkpoint would still be silent when the operator switched it off,
  and would be stale when the operator engaged it again between attempts.
  So the result says what is true when it is built: when an `Autofocus`
  exists and `is_engaged()` is false, `assumed` gains `the autofocus is off
  after the survey; the scan never switches it on: lock it on the sample
  before the scan`. Nothing is added when it is engaged or the stand has no
  autofocus (the demo and the fake do have one, switched off). The WARNING at
  the switch-off stays and the line in `assumed` made at the switch-off goes:
  one line, not two. The read at the end is made after the fit has passed
  its checks. `MicroscopeHaltedError` and `MotionStoppedError` propagate
  from it (FM-64); another `HardwareError` is a finding and not a lost
  survey: `the autofocus's state could not be read after the survey
  (<Type>: <first line>); check it before the scan`.
- **Unknown soft limits are labelled.** `limits_um()` is `None` on the
  demo, the fake and every stand whose profile has no `xy_soft_limits_um`
  (the Ti2-E's is the only one): `assumed` gains `the stage reports no soft
  limits, so the survey's moves were not checked against any travel`.
- **The order of `assumed`**: the limits line (cut points, or unknown
  limits), the skipped-points line, the autofocus line.
- **`margin_pitches` is a margin on each side, and the check is one pitch in
  all.** `survey_extent_note` asks for the wells' span plus one pitch, which
  is 0.5 of a margin on each side. Measured on the 96-, 12- and 24-well
  plates: with `jitter=0` the check passes from 0.50 and not at 0.48; with
  the default jitter it passes at 0.45 on every seed, and at 0.40 on all
  eight for the 96-well plate and 7 of 8 for the others, because the nudge
  moves the outermost points outwards. The parameter's description says: how
  far past the outer wells the survey reaches on each side, in pitches; a
  survey that spans less than the outer wells plus one pitch is refused,
  because A1 cannot be pinned down: about 0.5 here, a little less with
  jitter. The extent note's own text is a port and does not change.

## 7. Wells and the scan plan (#21)

`smc/plugins/scan_plan/wells.py`, `algorithm.py`, `plugin.py`, entry point
`scan-plan`. A port of `wells.py` and the geometry of `scan.py`, with
their tests, plus clipping to the well.

### 7.1 Wells

```python
def parse_wells(plate: str, text: str) -> tuple[str, ...]: ...
def order_wells(
    plate: str, wells: Iterable[str], *, serpentine: bool = True
) -> tuple[str, ...]: ...
```

The grammar of `wells.parse`: items separated by commas or spaces, `A1`,
`B2-B5` (a run in one row), `A1:D6` (a rectangle, corners in any order),
`C*` (a row), `*3` (a column), `all`; `a01` and `A 1` normalise to `A1`
(`normalise_well`, §3.2, `wells.normalise` ported there).
**Unlike the port**, an item that cannot be read or a well that is not on
the plate raises one `ValueError` listing every such item (the page's
regression "a nonsense range is reported, not silently ignored");
duplicates are removed. `order_wells` gives plate order by rows,
alternate rows reversed when `serpentine`; the result is what the scan
visits and the route follows.

Refined while planning #21: `B2-B5` and `A1:D6` are read alike, as the
rectangle of two corners (the port's behaviour, so `B2-D2` is a run down a
column); `all` is read in any case; a text that selects no well is a
`ValueError` ("no wells selected"). Rows and columns are found in
`plan.all_well_names` (§3.2), so `AA3` on a 1536-well plate is a well and
`C*` is the row whose names start with `C` and continue with a digit.

### 7.2 Grid and clipping

```python
def grid_size(
    fov_um: tuple[float, float],
    well_size_um: tuple[float, float],
    *,
    coverage: float,
    overlap: float,
) -> tuple[int, int]: ...
def plan_fields(
    plate: PlateResult,
    wells: Sequence[str],
    *,
    fov_um: tuple[float, float],
    footprint_of: Callable[[float, float], Sequence[tuple[float, float]]],
    rows: int,
    columns: int,
    overlap: float,
    clip_to_well: bool,
) -> tuple[useq.WellPlatePlan, tuple[PlannedField, ...]]: ...
def estimate_s(
    fields: Sequence[PlannedField], timings: TimingsResult, *, refocus: bool
) -> float: ...
```

- `grid_size` returns `(rows, columns)`, per axis `max(1, ceil(max(0,
  size × coverage − fov) / step) + 1)` with `step = fov × (1 − overlap)`
  (`fields_for_coverage`; rows from the heights).
- `plan_fields` builds `plate.to_plan(selected_wells=..., well_points_plan=
  GridRowsColumns(rows, columns, fov_width, fov_height, overlap=(100 ×
  overlap, 100 × overlap)))` with the wells in the given order and takes
  its `image_positions`. With `clip_to_well`, a field is kept when its
  footprint (`footprint_of(x, y)`, §3.1) intersects the well shape
  (`well_geometry`): for a round well, when the distance from the centre to
  the footprint polygon is below the radius; for a square well, when the
  footprint and the rotated square overlap (separating axes). A field
  that misses the well images only plastic. `index` is renumbered from 0
  in each well; `name` stays useq's.
- Refined while planning #21 (measured, useq 0.9.2): an image position's
  name is `<well>_<index:04d>`, a 1 × 1 grid included (`A1_0000`), so its
  well is the name up to the last `_`; a position whose well is not one of
  `wells`, or whose `x` or `y` is `None`, is a `ValueError` (useq changed,
  not the plate). The corners of a footprint turn one way or the other
  with the camera's mirror, so the round well's inside test accepts both
  orientations. Measured on the Ti2-E's assumed camera (1412 × 1412 at
  0.275 µm, 10 %): 305 of 361 fields of a 96-well well are kept, whether
  the distance is compared with `<` or `≤`; with a frame of 300 × 512 at
  1 µm on a plate turned by 13°, a 25 × 15 grid keeps 299 of 375 fields,
  the same set as a 40 × 40 sampling of each footprint tested with
  `WellGeometry.contains`.
- `estimate_s` sums, over the fields in order, `timings.move_ms(distance
  from the previous field)` (the first from the first field: 0),
  `refocus_ms` when `refocus` and known, and `snap_ms`, in seconds.

### 7.3 Plugin

```python
class ScanPlanParams(PluginParams):
    wells: str = Field(description="A1, B2-B5, A1:D6, C*, *3 or all.")
    serpentine: bool = True
    objective: str | None = None
    overlap: float = Field(0.10, ge=0.0, lt=0.9)
    coverage: float = Field(1.0, gt=0.0, le=1.5)
    rows: int | None = Field(None, ge=1, le=100)
    columns: int | None = Field(None, ge=1, le=100)
    clip_to_well: bool = True
    pixel_size_um: float | None = Field(None, gt=0.0, le=100.0)
```

`requires = (Camera,)`, `reads = ("plate",)`, `reads_optional =
("camera-calibration", "timings")`, `provides = "scan-plan"`. `rows` and
`columns` are both given or neither (a model validator); neither means
`grid_size`. Nothing moves, so it runs in a dry run like any other time.

- **The objective**: the parameter, else the turret's current one, else
  `None` (no turret). An objective that is not among `turret.labels()` is
  a `PluginRunError`, and so is one given on a stand without a turret
  (nothing could check it, §8).
- **The matrix**, in this order: (1) `pixel_size_um` given:
  `pixel_matrix`, `matrix_source="parameter"`, `assumed` gaining `pixel
  size given as a parameter; rotation 0 and no mirror assumed`; (2) a
  usable `camera-calibration` whose `objective` equals the plan's and
  whose `image_shape_px` equals `camera.image_shape()`: its matrix,
  `"camera-calibration"` (a calibration of another objective or shape is
  ignored, and `assumed` says why: `camera calibration not used: it is of
  <its objective>, the plan is for <objective>`, or `… it is of a frame of
  <its shape>, the camera gives <shape>`); (3) the stand's configured
  size, when finite and positive (FM-104): for the objective in place, or
  with no turret, `camera.pixel_size_um()`; for another objective,
  `turret.pixel_size_um(objective)`; then `pixel_matrix`, `"profile"`,
  `assumed` gaining `no camera calibration for <objective>: rotation 0 and
  no mirror assumed`. Since #383 (§23), a `calibration` parameter comes
  between (1) and (2), and the line of (2) for another objective names it.
  Refined while planning #21 (FM-103): the turret's size
  is the profile's **at binning 1** (`MMObjectiveTurret.pixel_size_um`),
  while the camera's includes the binning, the magnification factor and
  MMCore's own pixel calibration (m2 §8.10), so the turret's would plan a
  binned camera with fields half their size. Nothing in the `Camera`
  protocol reads the binning, so a plan for an objective that is not in
  place also gains `pixel size of <objective> from the profile, for binning
  1: the camera's binning is not applied`. On the demo (measured) the
  camera reports 1.0 µm with the 10× in place, the profile 0.65, the 40×
  0.1625 and `Objective-2` 0.0; (4) otherwise
  `PluginRunError`: "no pixel size for <objective>: run
  camera-calibration with it in place, or add it to the profile's [camera]
  pixel_size_um" (#21's "refuses to plan without a pixel size").
- `image_shape_px` is `camera.image_shape()`; `fov_um` the footprint's
  extent at the origin. When the camera's rotation makes the
  axis-aligned grid open gaps, `assumed` says so and suggests the overlap
  to use. Refined while planning #21: the angle that matters is `φ`, the
  distance from the calibration's `rotation_deg` to the nearest multiple of
  90° (a camera at 90° or 180° fills its bounding box; the unfolded
  `sin |θ|` warned at 90°); gaps are reported when `sin φ > overlap / 2`,
  and the overlap suggested is `ceil(200 sin φ) / 100`, or, when that is
  0.9 or more, none (the line says to straighten the camera). Measured on a
  square frame: at 10 % overlap the first gap opens at 3.8°, the threshold
  is 2.9°, so the warning comes early, never late. A matrix from the
  profile or a parameter assumes no rotation and never warns.
- `estimate_s` is `estimate_s(...)` when `timings` is usable, with
  `refocus` true when the stand has an `Autofocus`; else `None` (unknown,
  not guessed: the page says "run timing for an estimate").

**Tests**: the ported `test_wells.py` and the geometry of `test_scan.py`,
with `test_parse_rejects_and_lists_every_item_it_cannot_read`;
`test_a_96_well_at_40x_keeps_305_of_361_fields` (1412 × 1412 at 0.275 µm,
10 %); `test_clipping_a_square_well_follows_the_plate_rotation`;
`test_wells_are_visited_in_the_given_order_not_raster`;
`test_the_matrix_comes_from_a_calibration_of_the_same_objective_only`;
`test_no_pixel_size_refuses_to_plan`; `test_the_estimate_is_none_without_timings`;
a non-square camera (FM-73). **Failure modes**: FM-73, FM-104, FM-103.

## 8. The scan (#213)

`smc/plugins/scan/plugin.py`, entry point `scan`. A port of
`scan.run_iter`, redesigned around the layer.

```python
class ScanParams(PluginParams):
    autofocus: Literal["if-engaged", "required", "off"] = "if-engaged"
    max_failed_in_row: int = Field(3, ge=1, le=100)
    max_unlocked_in_row: int = Field(5, ge=1, le=1000)
    landing_tolerance: float = Field(0.05, gt=0.0, le=0.5)  # #228, below
```

```python
class ScanCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    token: str
    wells_done: tuple[str, ...]
    frames: tuple[ScanFrame, ...]
    failed: tuple[FailedField, ...]
```

`requires = (XYStage, Camera)`, `reads = ("scan-plan",)`, `provides =
"scan"`. Optional capabilities through `microscope.get`: `ZStage` (read
only), `Autofocus`, `ObjectiveTurret`.

**Before anything moves**, in this order (FM-30):

1. **The objective** (#14's turret): when the plan names one and the stand
   has a turret, `turret.current()` must equal it, else `PluginRunError`:
   "the objective in place is '<current>'; the scan was planned for
   '<planned>'. Turn the turret to '<planned>' (smc never turns it during a
   scan), or run scan-plan again with objective=<current>". No turret:
   `assumed` gains `the objective was not checked: the stand has no
   turret`.
2. **The camera**: `camera.image_shape()` equals the plan's (a binning
   change), else `PluginRunError`. `camera.pixel_size_um()` is noted as
   `p0`.
3. **The autofocus**: `required` with no autofocus or one not engaged:
   `PluginRunError` ("engage PFS on the stand before the scan").
   `if-engaged`: used when it is engaged now. `off`: never touched.
4. **The disk**: `shutil.disk_usage(run_dir).free` must exceed 1.1 × the
   fields left × `H × W × itemsize` of a frame (about 4 MB at 1412²,
   1.2 GB per 96-well well), else `PluginRunError` naming both numbers.

**Per well**, before its first field **and after its last** (FM-130): the
objective (when checked) and the pixel size are read again; a change of
objective, or a pixel size more than 1 % from `p0` (the magnifier lever,
FM-103), is a `PluginRunError`, the checkpoint keeping the wells done. A
well is marked done only when the stand was the same at both ends, so a
change in the middle of a well leaves that well out of the checkpoint.

**Per field**: `ctx.raise_if_stopped()`; `readback = xy.move_to_um(x, y)`;
a readback farther from the target than `landing_tolerance` of the plan's
smaller field side is a failed field, and nothing else is read or moved for
it (#228, below);
when the autofocus is used, not `is_locked()`, and either still engaged (a
lost lock) or switched off by the scan's own failed `engage()` (FM-129), one
`engage()` (the profile's lock timeout; it drives Z, which is what PFS is
for, as `nikon-control` engaged on every field); `focus` is `locked` or
`unlocked`, and `max_unlocked_in_row` unlocked fields in a row are a
`PluginRunError` (a lost lock over hundreds of frames must not read as a
scan); `z_um` from `ZStage.position_um()`, `pfs_offset` from
`offset_native()` when used; `frame = camera.snap()`; the frame is written
with `write_atomically` as a TIFF in its own dtype, as `smc snap` writes
it (`tifffile` into a `BytesIO`, with the `ScanFrame` as JSON metadata),
to `scan/<token>/<well>/<name>.tif`; `ctx.progress(done, total,
"<name>")`.

**Errors on a field**: `MicroscopeHaltedError` and `MotionStoppedError`
pass first (FM-64); any other `HardwareError`, `SafetyRefusedError`
included, records a `FailedField` (one line, `<Type>: <message>`) and goes
on; `max_failed_in_row` failures in a row are a `PluginRunError`. An
`OSError` writing a frame is not a field's fault (a full disk, a lost
share, a file held open): it ends the scan as a `PluginRunError` that names
the frame's file and says what to do, never as an unexpected `OSError`
(#228, below). A scan that ends with no frame because every field failed is
a `PluginRunError` too (#228).

**Checkpoint** after each well (`ctx.save_checkpoint`); a resume skips
`wells_done`, keeps the token and the frames. **Dry run**: no frame is
taken and nothing is written; the moves go through the layer's dry run,
and the result has no frames and `assumed` gaining `dry run: <n> fields
walked, no frame taken`.

The scan never moves Z, never turns the turret, never engages an
autofocus the operator left off, and returns to no start position.
*Amended by #341*: with `cell_focus` other than `"off"` (default `"off"`),
the scan moves Z in its focus on the cells, and switches off and engages
again the autofocus it uses, as §21.6 fixes.

**Refined while planning #213** (measured on the demo, 2026-10-28):

- **The plan's pixel size against the camera's**, before anything moves,
  after check 2 (FM-103). `p0` must be finite and positive to compare; when
  it is not, neither this check nor the per-well one runs, and `assumed`
  gains `the camera reports no pixel size: the plan's size and a change
  during the scan were not checked`. By `matrix_source`: `profile`, more
  than 1 % apart is a `PluginRunError` (the plan was made for an objective
  that was not in place, at binning 1, §7.3, or the binning or the
  magnification lever changed since); `camera-calibration`, more than 10 %
  apart (a calibration differs from the camera's report by its own
  correction, which camera-calibration warns about at 2 %; binning is 2×,
  the lever 1.5×); `parameter`, not compared, and `assumed` gains `the
  plan's pixel size was given as a parameter: not compared with the
  camera's`. Each refusal names both sizes and says to run scan-plan
  (and, for a calibration, camera-calibration) again with the stand as it
  is now.
- **The autofocus is used only when it is engaged and locked** before
  anything moves (#205: engaging drives Z to the stored offset's plane, so
  only a lock on this sample proves the offset is the operator's for it).
  `required` refuses an autofocus that is missing, off or not locked;
  `if-engaged` refuses one that is on and not locked ("lock it on the
  sample, or switch it off and run with autofocus=off"); `off` reads
  nothing from it. A failed `engage()` (one that returns False, or raises)
  leaves it off (the protocol); the next unlocked field engages it again,
  which is the scan's own state, not one the operator left. An autofocus the
  operator switches off during the scan is not engaged again (FM-129; its
  fields are `unlocked`). The scan adds no Z guard of its own around
  `engage()`: safety guards live in the layer, and that guard is #205's.
- **Paths are checked before anything moves.** Every frame's path is the
  text `scan/<token>/<well>/<name>.tif`; `well` and `name` must match
  `[A-Za-z0-9_-]+` and be no Windows device name, no two fields may share a
  path (compared without case), and each text goes
  through `resolve_in_run`; a failure is a `PluginRunError` (#228 adds the
  rules for names and the length). The
  checkpoint's `token` is constrained to `^[0-9a-f]{12}$` (`new_token()`),
  so a run document from elsewhere cannot point a write outside the run
  folder: `ctx.checkpoint` then raises `ResumeError`.
- **A resume** is a `PluginRunError` when `wells_done` names a well the
  plan does not hold, or a frame of the checkpoint has no file (a folder
  cleaned between attempts). A well partly done when the scan stopped is
  scanned again whole, its files rewritten.
- **The disk**: the itemsize is `camera.bit_depth()`'s, 1 byte up to 8
  bits, 2 up to 16, else 4 (the frame is not snapped to learn it); the
  need is `ceil(1.1 × fields left × H × W × itemsize)` and `free` must
  exceed it.
- **Per well**, `camera.image_shape()` is read with the objective and the
  pixel size. **Per field**, a frame whose shape is not the plan's ends
  the scan with a `PluginRunError` and is not written: every later frame
  would be wrong too.
- **Counts in a row** span wells and restart at 0 on a resume.
  `max_failed_in_row` counts failed fields, reset by a frame written;
  `max_unlocked_in_row` counts unlocked fields, reset by a locked one, and
  the field that reaches it raises before its snap. Reads per well that
  fail (`HardwareError`) are not a field's: they end the scan.
- **A move that timed out** on a stand that refuses `stop()` (m2 §11) is a
  failed field. The stage may still be running to its target; the layer's
  motion guard refuses the next moves while it is (`MotionInProgressError`,
  a failed field too), so such a stage fails the scan through
  `max_failed_in_row`. The scan never waits a motion out.
- **The dry run** runs the checks before anything moves, the disk's
  included, and walks the moves; it never snaps (`SampleCamera` ignores the
  dry run, measured) and never calls `engage()`. The demo's dry-run move
  returns its target and leaves the stage where it is (measured).
- A frame's TIFF carries `ScanFrame.model_dump(mode="json")` as
  `tifffile`'s JSON metadata, written into a `BytesIO` (measured with
  tifffile 2024.1.30 on Python 3.10 and 2026.9.20 on 3.13). `assumed`
  gains one line when the autofocus was engaged again (#228: it counts
  attempts and locks).

**Refined while planning #228** (measured on the demo, 2026-10-30, Python
3.13; the leftovers of #213, items numbered as in #228):

- **A scan with no frame is not `done`** (item 1). Reproduced: all 12
  fields of the demo plan failing with `max_failed_in_row=100` gave `done`,
  0 frames, 12 failed fields, and `cells` refuses such a scan. After the
  last well's checkpoint, when `failed` is not empty, no field of this
  attempt ended without an error and the checkpoint held no frame, the scan
  raises `PluginRunError`: `every field failed (<n>), the first <name>:
  <error>: check the stage and the camera, then run scan again without
  --resume (a resume skips the wells already walked)`. The checkpoint
  keeps every well, so a resume meets the same refusal: the message says
  so. A scan with one frame, from this attempt or from the checkpoint,
  stays `done` with its failures listed (the failures in a row are
  `max_failed_in_row`'s). A dry run is held to the same rule, its walked
  moves being the fields.
- **The readback is compared with the target** (item 2). The frame records
  the readback and `cells` maps from it, so a short landing does not
  misplace a cell: it leaves a gap in the coverage, or repeats a place.
  `landing_tolerance` is a fraction of the smaller side of the plan's
  `fov_um`, default 0.05, `0 < x <= 0.5`. Measured: the demo's readback is
  within 0.015 µm of the target on all 12 fields, and `fov_um` is (128, 96),
  so the default is 4.8 µm there. At the default overlap of 10 %, two
  neighbours that each land the tolerance away from each other, along their
  step, just keep touching: a landing within the tolerance leaves no gap on
  an axis-aligned grid. The distance is the `math.hypot` of the two axes
  and the test is `not distance <= tolerance`, so a readback that is not a
  number fails the field (FM-104). A field off target is a **failed field**:
  recorded with its target, error `StageOffTarget: the readback is <d> µm
  from the target, more than <t> µm (target (<x>, <y>), readback (<x>,
  <y>))`, one decimal each, counted by `max_failed_in_row`, and **nothing
  else is read or moved for it**: no autofocus call (an `engage()` drives
  Z), no snap. No retry and no wait (§8 above, m2 §11). Not checked in a dry
  run: the layer's dry-run readback is the target (measured), so nothing is
  measured. A plan whose smaller side of `fov_um` is not above 0 (an edited
  run document) is refused before anything moves: `the plan's field of view
  <fov> µm is not positive: run scan-plan again with the stand as it is now`.
- **A frame that cannot be written** (items 3 and 7).
  `write_atomically` raises `AtomicWriteError` for every `OSError`; it
  keeps the cause, and `kept` when only the rename failed. The scan turns
  it into a `PluginRunError`, raised outside the handler (FM-98). When
  `kept` is set and the cause is a `PermissionError` (Windows refuses to
  replace a file another program holds open, as a viewer on a well that a
  `--resume` rewrites): `the frame <file> cannot be replaced: another
  program holds it open (a viewer, a backup or antivirus agent): close it,
  then run scan --resume`. Any other: `the frame <file> could not be
  written (<strerror>): free some space if the disk is full, or check the
  network share, then run scan --resume`. `<file>` is the run-relative text,
  `scan/<token>/A1/A1_0001.tif`. The checkpoint keeps the wells done; the
  well in progress is scanned again whole. The retry of `write_atomically`
  stays 5 × 0.05 s: a viewer holds the file until it is closed, so a longer
  wait only delays the message, and the scan has no `time.sleep` of its own.
  The files are not tested for "held open" before the scan: opening a file
  does not show the condition that blocks its replacement, and the
  condition cannot be produced here. The absolute path of each frame is the
  one `_paths` resolved before anything moved, and the write does not
  resolve again, so no `OSError` of `resolve` becomes a `ValueError` at a
  field (measured: 24 `resolve_in_run` calls for 12 fields, 12 after). A
  `ValueError` whose cause is an `OSError`, from `resolve_in_run` in
  `_paths` (a share that drops during the check), is `_unreadable` there
  too, as in `_is_file_in_run` (FM-116).
- **MAX_PATH** (item 4). On Windows only (`os.name == "nt"`), before
  anything moves, the longest path the scan writes must be shorter than 260
  characters (MAX_PATH counts the terminating NUL). That path is the
  temporary file of `write_atomically`, `.<name>.<8 hex>.smc-tmp` beside the
  frame, 18 characters longer than the frame's own name (measured): its
  folder is then at least 24 characters under the 248 a directory may
  have, so one check covers both. `smc.run.files.temporary_path(path)` makes
  that name for `write_atomically` and for this check, so the two cannot
  differ. Refusal: `a frame's path would reach <n> characters while it is
  written, and Windows takes 259 at most unless long paths are enabled (not
  assumed): move the run document to a shorter folder, then run scan
  again`. Whether long paths are enabled is not read: the registry key does
  not say whether this Python honours it, and unknown beats guessed. A PC
  that has them enabled is refused too, and the fix costs a folder move.
- **Names** (item 8). `frame_path` refuses the Windows device names `CON`,
  `PRN`, `AUX`, `NUL`, `COM0`..`COM9`, `LPT0`..`LPT9`, in any case, for the
  well and for the field name alike (a part has no dot, so `NUL.tif` is the
  same device). The list is its own: `PurePath.is_reserved` is deprecated
  in 3.13 and removed in 3.15 (#203 holds the same rule for
  `resolve_in_run`; this one is the scan's own parts and stays). Two
  fields collide when their well and name agree after `str.casefold`:
  NTFS and the default APFS do not tell `A1_0001` from `a1_0001`. Measured
  on an APFS volume: a plan with `A1_0000` and `a1_0000` in one well was
  `done` with 12 frames in the result and 11 files on disk. The refusal
  names both frames' paths.
- **The autofocus flag** (item 6, FM-129). The scan's own failed `engage()`
  switches the autofocus off, and the flag that lets the next field engage
  it again (`off_by_the_scan`) was set after the call returned, so an
  `engage()` that raised left it unset (measured: one `engage()` call, then
  the scan ended with "lock it on the sample" for a switch-off nobody made).
  The flag is set before the call and is `not locked` after it. A second
  hole, found while planning: the flag also outlived a lock someone else
  took (measured: the scan's engage fails at field 2, the operator locks
  PFS by hand and switches it off again, and the scan engaged it a second
  time). The flag is cleared whenever a field reads the autofocus locked.
- **The `assumed` line** (item 9): `the autofocus was engaged again on <a>
  fields and locked on <l> of them`, when `a > 0`; an attempt is an
  `engage()` call started, one that raised included.
- **Closed, with a reason.** Item 5: `ScanResult.assumed` does not carry the
  plan's lines: the report gathers the `assumed` of `scan-plan`, `scan`,
  `cells`, `density` and `route`, each with its stage's name, so a copy
  would list the plan's lines twice, and `cells` needs none of them. Item
  10: the start check of a well stays after the end check of the one
  before. Skipping it saves three reads per well; the checkpoint write
  between the two is time in which a lever can move, and nothing measured
  what the reads cost on the Ti2-E.

**Tests of #228** (names in the plan on the issue; demo with `SampleCamera`,
the stage wrapper gaining a landing error by move number): every decision
above has one, and each is checked from both sides of its bound (a landing
of 4.0 µm passes and 6.0 µm fails a tolerance of 4.8 µm, on the x axis and
on the smaller side of a non-square frame, FM-73). **Failure modes**: FM-24,
FM-30, FM-38, FM-64, FM-88, FM-98, FM-104, FM-116, FM-129, FM-130.

**Tests** (demo with `SampleCamera` unless said): `test_a_scan_writes_one_tiff_per_field_and_records_the_readback`;
`test_the_wrong_objective_is_refused_before_anything_moves` (fake turret,
empty move log); `test_a_turret_turned_mid_scan_stops_at_the_next_well`;
`test_a_changed_pixel_size_stops_the_scan`; `test_one_failed_field_is_recorded_and_skipped`;
`test_three_failed_fields_in_a_row_fail_the_scan_with_the_checkpoint_kept`;
`test_a_halt_is_not_a_failed_field` (FM-64);
`test_a_stopped_scan_resumes_with_the_same_token`;
`test_an_unlocked_autofocus_is_engaged_once_per_field` (fake);
`test_too_little_disk_is_refused_before_moving` (patched `disk_usage`);
`test_a_dry_run_scan_writes_nothing`. **Failure modes**: FM-14, FM-12,
FM-64, FM-88, FM-30, FM-38, FM-24, FM-103, FM-111.

**Refined while planning #430** (measured on the demo, 2026-11-11, Python
3.13, useq-schema 0.9.2, tifffile 2026.9.20, pydantic 2.13.5; the owner asked
for a plate scanned at 10× and 40× "saving the images with the position of each
image on the well"). A frame's record placed it on the stage; the well's centre
was in another stage's record (`plate`, through `scan-plan`), so a TIFF read
alone, or a run folder copied without thinking of the plan, could not say where
in its well a frame lay. Measured: on the demo's seeded 96-well plate, with a
1 × 1 grid, the three fields lie at the three wells' centres and `readback −
well_geometry(plan.plate_plan, well).center_um` is `(0.0, 0.0)` for A1 (0, 0),
A2 (9 000, 0) and B2 (9 000, −9 000) µm; with 2 × 2 fields it is `(±57.585,
±43.2)`; and `ScanFrame` today refuses a key it does not know, so a frame that
carries the new keys is not readable by an smc without them (FM-107). A
prototype of the model (two optional floats, `cell_focus`'s wrap serializer
extended, a validator that refuses one key without the other) dumps a frame
without them exactly as today, reads its own dump back equal, and puts the keys
in the TIFF's JSON as written. The report's two ways to place a footprint,
`footprint(x, y) − centre` and `footprint(x − cx, y − cy)`, differ by at most
3.6e-12 µm on 20 000 random frames up to ±50 mm from the origin, and print the
same number at 0.1 µm for all 160 000 corners.

- **The record.** `ScanFrame` gains, last and after `cell_focus`, `well_dx_um:
  float | None = None` and `well_dy_um: float | None = None`: the frame's
  centre minus the well's centre, **in stage axes and µm**, taken from the
  readback (`x_um`, `y_um`), not the target: a frame that landed 3 µm right of
  its field reads 3 µm more. `(0, 0)` is the well's centre; a positive `dx` is
  the stage's +X side of it. They are **not turned into the plate's axes**: on
  a plate turned by θ a frame one pitch along a row of wells reads `(p cos θ,
  p sin θ)`, which is what the stage did; `WellGeometry.to_plate_frame` makes
  the other reading from them. `None` means *not recorded* (a scan written
  before #430), never zero. The two go together: a validator refuses one
  without the other, and both are finite like every float of a slot value
  (FM-32, FM-104).
- **In the TIFF and the document.** `encode_frame` already writes
  `ScanFrame.model_dump(mode="json")`, so the keys are in each TIFF's JSON with
  no code of their own. As for `cell_focus`, a missing value is **left out of
  the dump**: a frame without them is written exactly as before, so a scan of an
  older smc reads and writes back unchanged, and `SCHEMA_VERSION` stays 1. A
  frame *with* them is refused by an older smc (it refuses a key it does not
  know rather than drop it, FM-107): upgrade the smc that reads it.
- **The well's centre** is `well_geometry(plan.plate_plan, well).center_um` of
  the plan the fields came from: the call `plan_fields`' clipping made (§7.2),
  in the frame of the plan's objective (§18.3, the translation of a plate moved
  between objectives included). It is **computed once per scanned well before
  anything moves** and not re-derived: the scan does no pitch or rotation
  arithmetic of its own. A well that has fields in the plan and that its plate
  does not hold is refused with nothing moved, after the path check (an edited
  run document: measured, today's scan photographs a field of a listed well
  `Z9` that no plate holds, and the report then fails on it):
  `the plan has fields in well '<w>', which its plate does not hold: run
  scan-plan again with the stand as it is now`. A well with no field needs no
  centre and is not looked up. The offsets are made from the
  readback at the field, after the move and the landing check, in the same
  statement that builds the frame's record; a failed field has no frame, so no
  offset (`FailedField` keeps the target), and a dry run takes no frame.
- **`assumed` says what the offsets rest on**, when frames are written (never
  in a dry run), after every line the scan already writes, from the plan alone
  (the scan reads `scan-plan` and no other record):
  - `objective_shift_um is None` (an older plan, an objective in use that is not
    known, or a plate that records no objective or whose offset to this
    objective was not measured, §18.2 rules 2 to 4): `the frames' offsets from
    their well's centre (well_dx_um, well_dy_um) are against the plate as the
    plan holds it, registered with <plate_objective, or "an unknown objective">
    and not moved to the frame of <objective, or "the objective in use">: they
    carry any difference between the two (scan-plan's assumed says why)`.
  - a shift that is not `(0.0, 0.0)` (the plate was translated, §18.2 rule 5):
    `the frames' offsets from their well's centre (well_dx_um, well_dy_um) are
    against the plate moved from <plate_objective, or "an unknown objective">'s
    frame to <objective>'s by (<sx>, <sy>) µm, the profile's offsets between
    objectives`, the shift to one decimal.
  - `(0.0, 0.0)`, which is every plate registered with the objective in use, and
    every stand with no turret: no line.
  - frames kept from a checkpoint that an older smc wrote carry no offsets, and
    **the scan does not fill them in**: the TIFFs hold what was written, and a
    record that disagreed with its file would be worse than a gap. When `n > 0`
    of the `m` frames kept lack them: `<n> of the <m> frames kept from the
    checkpoint were written before a frame's offset from its well's centre was
    recorded: they carry no well_dx_um or well_dy_um`.

  The plate's own method (a seed, a lattice fit) is **not** said here: the plan
  does not carry it, the scan reads `scan-plan` only, and it is the plate
  record's to say (m5 §15 D6 draws it on the map). What the owner's request
  called a plate that was estimated or translated is what the plan knows: the
  shift above.
- **The report draws from them** (§13): a frame's footprint is drawn at
  `(well_dx_um, well_dy_um)` of the well's centre when the frame has them, and
  at the readback minus the well's centre when it has not (a scan of an older
  smc), the same page to the digits it prints. It does not compare the record
  with the plan's plate: a plan run again makes the scan stale, and the runner
  refuses a stale read, as it does for `x_um` (FM-111).
- **Not here.** `cells`, `density`, `route` and `nis-export` read stage
  coordinates, which they need, and stay as they are. The `scripts/dev` tools of
  #343 compute the offset from the plate record today and read the keys where
  they exist in their own change. The offsets are not filled into a scan that
  has none (a rewrite of its TIFFs), and the well's centre is not recorded
  beside them (it is in the plan).

**Tests of #430** (names in the plan on the issue): the offset of a frame
planned at a well's centre is `(0, 0)` and that of a frame that landed off is
the readback's, each on a well whose centre is not the origin (A2, B2: a scan
that forgot to subtract the centre reads `(9 000, …)`), on a non-square frame
and on a plate turned by 13°, where the stage's axes and the plate's differ
(FM-73); the keys are in the result and in each TIFF's JSON and agree; a record
without them loads and is written back without them; one key without the other
is refused; each `assumed` case, from both sides, and no line for the
objective in use; a resume across the older smc; the report's page the same with
and without the keys, and a frame whose offsets say otherwise drawn where they
say (it draws from the record). **Failure modes**: FM-32, FM-73, FM-104,
FM-107, FM-111, FM-124.

**Refined while planning #416** (PFS across a well's wall; merging this
decides it; measured on the Ti2-E fake, `main` 8549b4f, 2026-11-11). F11
(Ti2-E session 3, m2 §12.11.1): at the 40×, the plane the offset's floor
names lay 46.9 µm below the focus in C4 and at least 56.8 µm above it in
C3, while the two foci by eye lie 11.7 µm apart: relative to the focus, one
offset's plane moved by at least 104 µm between two wells. The scan moved
from well to well with PFS locked and engaged again only when a field read
unlocked, so PFS locked and following its reference over a wall was outside
every guard (m2 §12.2 D2, *Not chosen*: no watchdog between actions).
Measured: PFS locked, an XY move of one pitch with the plane 60 µm toward
the sample leaves PFS locked and Z +60 µm, with no word. With PFS switched
off before the move and `engage()` at the operator's offset (a seen offset)
after it, a plane at +40 µm or −45 µm locks under D2's band (`E` = 50 µm),
and one at +60 µm or −60 µm trips `AutofocusExcursionError`, switches it
off and forgets the offset, so the next `engage()` is refused. Z does not
move while PFS is switched off (0.00 µm on the fake; §12.11.1 on the
stand).

- **PFS never crosses a well's wall locked.** When the scan uses the
  autofocus (on and locked at the start, above), before the first field of
  every well, after the well's start check (FM-130) and not in a dry run,
  the scan reads `is_engaged()`; on, it calls `disengage()` and reads
  `is_engaged()` again. The move to the well is then made with PFS off and
  Z where PFS held it at the last field. The well's first field reads it
  unlocked and engages it again as the scan's own switch-off
  (`off_by_the_scan`, FM-129): one `engage()` at the operator's offset,
  under D2's band around where Z started, `z_start ± E`, which is where PFS
  held Z in the well before.
  - **The first well of an attempt too**: nothing tells in which well the
    operator locked, and a relock where they locked moved Z by −0.04 µm
    (m2 §12.11.1).
  - **At every objective and on every stand**, a rule of the scan with no
    parameter: it costs one engage per well (the Ti2-E's `settle_s` is
    1 s). Whether a plane moves between wells at the 10× is not known
    (m2 §12.11.3's probe has not run); a trip there says that the frames
    would have been far from the focus.
  - **An autofocus that is off at a wall stays as it is**: the operator's
    switch-off is not undone (FM-129), and the scan's own failed engage and
    a focus step's switch-off (§21.6) are already the scan's to engage
    again. Nothing is counted for that well.
  - **Not in a dry run**: `disengage()` acts in a dry run (m2 §3.2), and a
    dry run never calls `engage()`.
  - **A switch-off that cannot be made or read back ends the scan** before
    anything moves to the well, since PFS may still be on (unknown counts
    as on, FM-19): a `HardwareError` from any of the three calls is a
    `PluginRunError`, raised outside the handler (FM-98): `the autofocus
    could not be switched off before well <w> (<Type>: <message>): nothing
    moved to it, and PFS may still be on: switch it off at the stand, then
    run scan --resume`; `is_engaged()` still `True` after `disengage()`:
    `the autofocus still reads on after it was switched off before well
    <w>: nothing moved to it: switch it off at the stand, then run scan
    --resume`. `MicroscopeHaltedError` and `MotionStoppedError` pass first
    (FM-64). The checkpoint keeps the wells done.
- **A trip ends the scan.** An `AutofocusExcursionError` from a field's
  `engage()` (a wall's, a lost lock's, or the one after a focus step) is no
  longer a failed field: the layer switched the autofocus off and forgot
  its offset (m2 §12.2 D2), so every later engage would be refused and the
  scan could only fail `max_failed_in_row` fields with no frame and a
  message that blames the stage. It is a `PluginRunError`, raised outside
  the handler (FM-98): `the autofocus moved Z out of its bound while it
  locked at <field> (well <w>): <the layer's message, one line>; the
  checkpoint keeps <n> wells. Its offset names another plane here than
  where it held (m2 §12.11): with PFS off, focus on the sample in well <w>
  by eye and lock it from below (m2 §12.10), then run scan --resume (the
  well is scanned again whole)`. The engage after the last field (§21.6)
  keeps its rule: said, not a failure.
- **Counted apart.** An engage counts as a wall's when it is the first
  attempt after the scan switched the autofocus off at that well's wall (a
  field off target leaves it to the next field); a field that reads locked
  first ends the count's claim. §8's line keeps the engages after a lost
  lock and after the scan's own failed engage; the engages after a focus
  step stay the focus's (FM-165). `assumed` gains, after §8's line, when
  the scan switched the autofocus off at a wall at least once: `the
  autofocus was switched off before the first field of <w> wells, so that
  no lock crossed a well's wall, and engaged again after it on <a> fields,
  locked on <l> of them`.
- **The resume refusal** above names the new cause: `the scan being resumed
  used the autofocus, which is not locked now (the scan switches it off
  between wells, and a stop during the focus on the cells leaves it off):
  lock it on the sample, then run scan --resume`.
- **The 40× with this** (replaces m2 §12.11.3 step 3's "one well per run"
  for the scan): lock PFS from below in the first well (m2 §12.10 D7), then
  scan the wells in one run. A trip ends the scan at the well whose plane
  moved; that well is imaged once the operator has locked in it from below
  by hand and run `scan --resume`. Each well's relock is bounded by `E`: at
  50 µm, F11's jump would be read at about 57 µm, as it was in F11. The
  stand's copy may set `max_engage_excursion_um` lower for a 40× scan, as
  m2 §12.11.3 step 2 did, wide enough for a plate's change of height
  between wells (C3's and C4's foci by eye 11.7 µm apart), and back to 50
  before a 10× scan: the owner's call at the stand.
- **Not chosen**:
  - *Lowering Z before the move* (#416's first example). D2's band is
    centred on where Z starts: lowered by 200 µm, the way back up to the
    focus lies outside it, and every well would trip. With PFS off Z does
    not move, and the bottom's height changes by about 12 µm between two
    wells (the foci by eye), not 200.
  - *A first lock from the floor in each well*. Its premise is a focus by
    eye at that place (m2 §12.10 D5), which a scan has not; in F11's C3 the
    floor's plane lay above the focus, where a first lock trips. The
    operator's first lock by hand stays the way into a well where the
    relock tripped.
  - *A bound per objective* (`E` per objective, m2 §12.11 D8's *Not
    chosen*). One `E` bounds every engage, and the stand's copy can be set
    for a session; a key per objective waits for numbers: each frame records
    its Z and focus, so the scan's run document says how far a well's lock
    lies from the last well's at each objective.
  - *Refusing runs of several wells at an objective the profile names*.
    Each well's relock is bounded like any engage, and the trip stops the
    scan at the first well whose plane moved; a refusal would make every
    well a run of its own at the 40× for good.
  - *Keeping the lock across walls under a watchdog*: m2 §12.2 D2's *Not
    chosen* (a thread of its own, M5).
- **Not here**: `goto-well` and the page's stage moves carry a lock over a
  wall the same way (#454). NIS-Elements' own acquisition from
  `nis-export`'s positions engages PFS at each position's offset, outside
  smc.

**Tests of #416** (names in the plan on the issue; the demo rig with the
recording `_Autofocus` fake, three wells of four fields): the switch-off
before each well's first field and its engage there, counted apart, and the
first well's too; an operator's switch-off at a wall not undone; a focus
step's switch-off not counted; nothing in a dry run; a switch-off that
raises, a state read that raises, and one that still reads on, each ending
the scan with nothing moved to the well; a trip at a well's first field
and one after a lost lock ending the scan with the checkpoint kept and no
later engage; the resume refusal's text. **Failure modes**: FM-19, FM-64,
FM-98, FM-129, FM-130, FM-165, FM-215.

## 9. Cells (#23)

`smc/plugins/cells/overlap.py`, `detector.py`, `algorithm.py`,
`plugin.py`, entry point `cells`. The detector is a port of `detector.py`
and the in-frame suppression a port of `overlap.py`; the de-duplication
across fields is new.

### 9.0 What was measured (2026-10-27, planning #23)

On `local/models/cell_classes_model-cat-v2.pth` with torch 2.13 and
torchvision 0.28 (Python 3.13, Apple M5 Pro, CPU), and on
`smc.testing.synthetic` (scripts in the design session's scratchpad):

- The checkpoint's `classes` is a `list`; `roi_heads.box_predictor.cls_score.weight`
  has 4 rows; `load_state_dict(strict=True)` matches every key. Building
  and loading take 0.22 s, the SHA-256 of the 166 MB file 0.06 s, one
  1412² noise frame 0.32 s after a first call of 0.49 s; noise and a
  constant frame give no detection. The output is `boxes` `float32`
  `(n, 4)`, `labels` `int64`, `scores`. The machine has MPS, no CUDA.
- `torch.load(weights_only=True)` of a file that is not a checkpoint
  raises `_pickle.UnpicklingError`; `load_state_dict(strict=True)` of a
  partial state raises `RuntimeError`. A checkpoint of a few tensors
  (14 KB) is enough to test every refusal that precedes the build.
- `import torch` with `sys.modules["torch"] = None` raises
  `ModuleNotFoundError`, a subclass of `ImportError`.
- `tifffile.TiffFileError` derives from `ValueError`.
- **mypy**: a module that imports torch and torchvision inside a function
  passes `mypy --strict` with and without torch installed under an
  override `module = ["torch", "torch.*", "torchvision", "torchvision.*"]`,
  `ignore_missing_imports = true`, `follow_imports = "skip"`. Without it,
  mypy fails both ways: `import-not-found` without torch, `import-untyped`
  for `torchvision.models.detection` with it.
- **Pixels to stage**: on a `(200, 320)` frame at 1 µm per pixel, two
  fields 288 pixels apart along the frame's columns, blobs at whole pixels
  of the first frame and a thresholding stub whose box edges are the
  extreme pixel indices of each blob, `to_cells` puts every blob at its
  stage position within 1e-4 µm at 0°, 30°, 30° mirrored and −12°
  mirrored, and the merge leaves each blob once. Swapping height and width
  moves the worst cell 84.85 µm; the half-pixel centre `W / 2` moves it
  0.71 µm.
- **Merge**: single-linkage over the cells of a well chains distinct
  cells. Frame 0 sees two cells 6 µm apart, frame 1 one box 3 µm from
  each: the union-find of the first version of this section removed two,
  and kept one cell for two. Two cells 5.5 µm apart seen by both frames
  with a 0.8 µm calibration error: it removed three of four. The rule of
  §9.2 removes one and two.

### 9.1 The detector (decides §1.13)

```python
@dataclass(frozen=True, slots=True)
class Detection:
    row0_px: float
    col0_px: float
    row1_px: float
    col1_px: float
    label: str
    score: float


@runtime_checkable
class Detector(Protocol):
    classes: tuple[str, ...]
    device: str

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]: ...
```

A protocol, and no discovery in M3: one detector, `TorchDetector`, and
the tests' stubs. `classes` are the names a detection can carry, the
background excluded; `device` is the device it runs on, resolved (`cpu`,
`cuda`, `mps`, never `auto`), which `CellsResult.device` records.

```python
DEFAULT_MODEL = "cell_classes_model-cat-v2.pth"


def normalize_plane(
    plane: npt.NDArray[Any], percentiles: tuple[float, float] = (0.5, 99.5)
) -> npt.NDArray[np.float32]: ...


class TorchDetector:
    def __init__(
        self, weights: Path, *, device: str, score_threshold: float
    ) -> None: ...


def default_model_path() -> Path: ...
def load_detector(
    model_path: Path | None, *, device: str, score_threshold: float
) -> Detector: ...
```

- `normalize_plane` is numpy only (float32, clipped to the 0.5 and 99.5
  percentiles, scaled to [0, 1] with `hi − lo + 1e-6` below, so a
  constant frame gives zeros).
- `TorchDetector` imports torch and torchvision **inside** `__init__`
  (the plugin module imports without them, so `smc run --list` shows
  `cells` on every PC). It reads the checkpoint with
  `torch.load(weights, map_location="cpu", weights_only=True)` and refuses
  with a `ValueError` naming the file, in this order, before anything is
  built: a file `torch.load` cannot read (any exception of it, chained);
  a checkpoint that is not a `dict` with a `dict` under
  `model_state_dict`; one without a `list` or `tuple` `classes` of at
  least two strings; `classes[0]` other than `__background__`; a state
  without `roi_heads.box_predictor.cls_score.weight` or whose number of
  rows is not `len(classes)`. It then resolves the device, builds
  `fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
  num_classes=len(classes))`, loads the state with `strict=True` (a
  `RuntimeError` becomes a `ValueError` naming the file), `eval()`, moves
  it to the device. `classes` is `tuple(classes[1:])`.
- `device`: `auto` is `cuda` when available, else `cpu` (never `mps`
  unasked: the port's caution, though MPS ran in §2.1); `cuda` or `mps`
  asked for and not available is a `ValueError` that says to use `auto` or
  `cpu`, never a silent fall-back (the port printed a warning and ran on
  the CPU).
- `detect`: refuses a frame that is not 2-D (`ValueError`);
  `normalize_plane`, the plane repeated to 3 channels, one frame per
  forward pass under `torch.no_grad()`, the model's own resize kept; boxes
  from torchvision's `[x0, y0, x1, y1]` to `Detection`;
  `score < score_threshold` dropped; `label` is the checkpoint's
  `classes[label_id]`.
- `default_model_path()`: `$SMC_MODELS / DEFAULT_MODEL` when the variable
  is set and not empty or blank (FM-78), else
  `<checkout>/local/models/DEFAULT_MODEL` with the checkout
  `Path(smc.__file__).resolve().parents[2]` (the editable install of every
  PC; in a worktree, the worktree).
- `load_detector` resolves `None` to `default_model_path()` and raises
  `PluginRunError`, in this order: a path that is not a file (naming it,
  and `SMC_MODELS`); torch or torchvision that cannot be imported, an
  `ImportError` ("install the detect extra: pip install -e .[detect]");
  an `OSError` while importing them (a DLL that does not load on
  Windows: "torch is installed but did not load (…); reinstall the detect
  extra"); and every `ValueError` of `TorchDetector` (a refused checkpoint,
  an unavailable device), with its message. The file is checked first so
  that a PC without torch still says where it looked.

**The blob detector (#260).** The trained model does not see the
synthetic cells of the demo-sample stand (M1 §8, "The demo-sample
stand": at most 2 `single` among 14 objects), so the workflow could not
be followed on images past the scan. A second detector, in the same
module, numpy and scikit-image only:

```python
BLOB_DETECTOR = "blobs"  # CellsResult.model when it ran
BLOB_DETECTOR_IDENTITY = (
    "smc BlobDetector 1: p75 + 8 sigma, 9 px to 5 % of the frame, "
    "axis ratio >= 0.3, single from 0.75"
)  # its SHA-256 is CellsResult.model_sha256


class BlobDetector:
    classes = ("single", "doublet")
    device = "cpu"

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]: ...
```

`detect` refuses a frame that is not 2-D (`ValueError`), as
`TorchDetector` does. On the frame as `float64`: the background is its
75th percentile and the noise `σ = 1.4826 × median(|Δ|) / √2`, `Δ` the
differences of neighbouring pixels along a row (a well wall is a step,
which a spread around the median would count as noise); `σ` not finite or
not positive gives no detection. The mask is `frame > background + 8 σ`,
its connected components `skimage.measure.label` (8-connected, the
default) and `regionprops`. A component is dropped when its area is under
9 pixels or over 5 % of the frame (a sliver of well seen past a wall), or
when `axis_minor_length / axis_major_length` is under 0.3 (a strip) or the
major axis is 0; it is a `single` from a ratio of 0.75 up, else a
`doublet`. Its box is its `bbox` as floats, its score 1.0. **Measured
2026-10-29** (scikit-image 0.26.0) on `PlateSample` frames of 5 single and
3 paired spots of radius 5 µm, `CELL_INTENSITY` above the well: exactly 5
`single` and 3 `doublet` at 0.1625, 0.25, 0.5, 1.0 and 2.0 µm per pixel;
no detection on 21 cell-free frames across a well wall (centre from −½
to +½ frame past it) nor at a corner, at every one of those pixel sizes.
With the median for background and its spread for `σ` (the first
prototype), a frame centred on the wall gave 5 false cells at 1.0 µm.
With 2 % for the largest area, the doublets vanished at 0.1625 µm. It
knows nothing of cells: it is for the synthetic sample, and the result
says so.

### 9.2 Algorithm

`overlap.py` is `nikon-control`'s `overlap.py`, typed, with its 17 tests
(`tests/unit/test_overlap.py`): `bbox_area`, `bbox_intersection`,
`bbox_iou`, `bbox_iomin`, `OverlapCandidate`, `select_survivors`,
`count_overlaps`, `PREFER_AREA`, `PREFER_SCORE`, `DEFAULT_MAX_OVERLAP`.
Boxes are `(row0, col0, row1, col1)`, as `Detection` and `Cell.box_px`.

```python
def suppress_in_frame(
    detections: Sequence[Detection], *, max_overlap: float = 0.7
) -> list[Detection]: ...
def to_cells(
    detections: Sequence[Detection],
    frame: ScanFrame,
    frame_index: int,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> list[Cell]: ...
def edge_distance_px(
    box_px: tuple[float, float, float, float], image_shape_px: tuple[int, int]
) -> float: ...
def merge_across_fields(
    cells: Sequence[Cell],
    *,
    image_shape_px: tuple[int, int],
    radius_um: float,
) -> tuple[tuple[Cell, ...], int]: ...
```

- `suppress_in_frame` is `select_survivors` with `bbox_iomin` (intersection
  over the smaller box) and `prefer="score"`: the model's per-class NMS
  returns one object as both `single` and `doublet`, and the more
  confident label wins. The survivors come back **in input order**
  (`select_survivors` returns them in rank order).
- `to_cells`: the box centre `((col0 + col1) / 2, (row0 + row1) / 2)`
  through `stage_of_pixel` (§3.1), with the frame's `x_um`, `y_um`. The box
  is taken in the pixel-index convention of §3.1 as the detector gives it:
  whether the training annotations put a pixel's centre at its index or
  half a pixel further is not recorded, and it is at most 0.14 µm at 40×.
  `Cell.box_px` is the box as given, `Cell.frame` is `frame_index`.
- `edge_distance_px`: how far the box centre is from the nearest edge of
  its frame, the edges being the outer edges of the edge pixels (−0.5 and
  `W − 0.5`, §3.1): `min(cc + 0.5, W − 0.5 − cc, rc + 0.5, H − 0.5 − rc)`.
- **`merge_across_fields`** (new): cells of **one well** seen by
  **different** frames whose centres are at most `radius_um` apart
  (`<=`) are views of one object. The cells are taken best first: the
  largest `edge_distance_px` (the least cut view), then the higher score,
  then the lower frame index, then the input order. Each cell not yet
  removed is kept, and it removes, **from each other frame, at most one**
  cell not yet decided: the nearest within `radius_um` (ties by input
  order). A cell kept is never removed, any label merges with any label,
  and two cells of the same frame never remove each other (the in-frame
  step did that). Neighbours are found through a grid hash of cell
  `radius_um` keyed by well, so a well of thousands of cells is not
  quadratic. It returns the kept cells in input order and the number
  removed. `radius_um == 0` merges nothing (and divides by nothing).
  - Single-linkage (union-find over the pairs) was the first version and
    is rejected: it chains distinct cells through a third view (§9.0).
  - Ownership by nearest frame centre was rejected: with a calibration
    error near the boundary, both frames can disown the same cell.
  - A cell seen by the four fields of a corner is kept once: the best
    view removes one cell from each of the three others.

### 9.3 Plugin

```python
class CellsParams(PluginParams):
    # blobs: §9.1, for the demo-sample stand only (#260)
    detector: Literal["model", "blobs"] = "model"
    model_path: Path | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    score_threshold: float = Field(0.5, ge=0.05, le=1.0)
    max_overlap: float = Field(0.7, gt=0.0, le=1.0)
    merge_radius_um: float = Field(5.0, ge=0.0, le=100.0)


class CellsCheckpoint(BaseModel):
    token: str
    wells_done: tuple[str, ...]
    model_sha256: str
    merged: int = Field(ge=0)
```

`requires = ()`, `reads = ("scan",)`, `provides = "cells"`. In order:

1. The scan: `ctx.upstream("scan", ScanResult)`; a scan with no frame (a
   dry run, or every field failed) is a `PluginRunError`.
2. The model: `path = params.model_path or default_model_path()`, the
   detector through the module-level `load_detector(path, …)` (tests
   replace it with `monkeypatch.setattr` on `smc.plugins.cells.plugin`),
   then the SHA-256 of `path`, read in 1 MiB chunks (an `OSError` is a
   `PluginRunError` naming the file).
3. Resume: `ctx.checkpoint(CellsCheckpoint)`. A checkpoint whose
   `model_sha256` differs from the weights' is a `PluginRunError` ("the
   weights changed since the run being resumed; run cells without
   --resume"): cells of two models are not one result. Its `token` is kept;
   the cells of `cells/<token>.partial.json` (a `CellsFile`) whose well is
   in `wells_done` are kept and the others dropped (a crash between the
   partial file and the checkpoint). A partial file that is missing or
   does not validate is a WARNING, and every well is done again under the
   same token, `merged` from 0. Without a checkpoint, a new token.
4. Per well of `scan.frame_index()`, in its order, skipping `wells_done`;
   per frame: `ctx.raise_if_stopped()`; `load_frame` (a
   `FileNotFoundError`, a `ValueError` — a path out of the folder, or
   `tifffile`'s error — or an `OSError` is a `PluginRunError` naming
   `frame.file` and saying to run `scan` again); a frame that is not 2-D,
   not of `scan.image_shape_px`, or not finite is a `PluginRunError`
   naming it (the matrix of §3.1 is only right on that shape, FM-73);
   `detect`, timed with `time.perf_counter`; a label not in
   `detector.classes` is a `PluginRunError` (a detector bug, and the counts
   would not add up); `suppress_in_frame`, `to_cells` with the scan's
   matrix and shape; `ctx.progress(done, len(scan.frames), "<well> <name>")`
   where `done` counts the frames of the wells resumed too. Then
   `merge_across_fields` on the well; the partial file is rewritten with
   every cell so far (`write_atomically`), and only then
   `ctx.save_checkpoint` with the well added and its `merged` summed.
5. The end: `cells/<token>.json` written (`write_atomically`, JSON
   `ensure_ascii=True`, `allow_nan=False`, compact), then the partial file
   removed through `retry_on_permission_error` (missing is fine; a failure
   is a WARNING naming it, not a failed stage: the result is whole).

`CellsResult`: `file` the final file relative to the run folder; `model`
`path.name` (never the path, FM-50); `model_sha256`; `classes` and
`device` the detector's; `score_threshold`, `merge_radius_um` the
parameters; `frames = len(scan.frames)`; `counts` for every well of
`scan.frame_index()` and every label of `classes`, zeros included;
`merged` the sum over every well, the resumed ones through the
checkpoint;
`seconds_per_frame` the mean `detect` time of this attempt, `None` when
it detected no frame. `assumed` names the merge radius as chosen.

`detector` (#260): `model` loads the weights as above; `blobs` builds
`BlobDetector()` and never touches `model_path` or torch. `CellsParams`
refuses, with a `ValueError` naming the fix, `detector=blobs` with a
`model_path` ("model_path is for detector=model") or with a `device`
other than `auto` or `cpu` ("the blob detector runs on the cpu"). With
`blobs`, step 2 hashes `BLOB_DETECTOR_IDENTITY` (UTF-8) instead of a file,
so a resume across a change of its rules is refused like a change of
weights; `CellsResult.model` is `BLOB_DETECTOR`, `device` `cpu`, and
`assumed` adds "cells were found by the blob detector (detector=blobs),
which counts bright round spots for the synthetic plate of the
demo-sample profile; on a real sample its counts are not cells".

`merge_radius_um = 5.0` is **chosen, not measured**: half the size of a
yeast cell, well above the calibration's error, well below two cells
side by side. #198 looks at merged pairs on real frames.

**Tests** (stub detector unless said):
`test_a_cell_seen_by_two_overlapping_fields_is_counted_once`;
`test_the_least_cut_view_is_kept`; `test_two_cells_of_one_frame_are_never_merged`;
`test_a_box_between_two_cells_of_another_frame_removes_only_one` (§9.0's
chain); `test_merge_keeps_cells_of_different_wells_apart`;
`test_a_rotated_mirrored_matrix_puts_cells_at_their_sample_position` (a
`PlateSample` with blobs, a thresholding stub, FM-73);
`test_the_plugin_imports_without_torch` (a subprocess with
`sys.modules["torch"] = None`); `test_a_missing_model_says_where_it_looked`;
`test_a_resume_skips_the_wells_done`;
`test_the_trained_model_loads_and_finds_nothing_on_noise` skipped
without torch or the weights (never in CI). **Failure modes**: FM-73,
FM-78, FM-50 (no full path), FM-113, FM-24, FM-38.

### 9.4 The model on a stand's frames (#342)

The model's detections are checked on real frames before a scan's counts
are used: a reference set annotated by the owner, a script that prints
precision and recall per class for the model and for `blobs`, and three
pre-processing parameters of `cells` set from that measurement. This
section fixes the parameters, the measurement's code and the script's
files; the numbers come from session 3's frames, after the code.

**What was measured (2026-11-09, planning #342)**, with torch 2.13.0 and
torchvision 0.28.0 (`../nikon-control/.venv`, Apple M5 Pro, CPU):

- The checkpoint holds `classes` `['__background__', 'single', 'doublet',
  'debris']`, `epoch` 6, `val_mAP@0.5` 0.867, `init_from`, and
  `trainable_layers` 1, and **no pixel size, frame size or camera**.
  `nikon-control` trains and infers with the same pre-processing
  (`detector.normalize_plane`, percentiles 0.5 and 99.5, the plane repeated
  to three channels, full frames, no crop, no `min_size` or `max_size`, so
  torchvision's 800 and 1333; dihedral flips as the only augmentation). Its
  only ND2 (`doublet003_xy05.nd2`, untracked) is a **Prime 95B** frame of
  1412², 16 bits, 0.27555 µm per pixel, under a 40× λD NA 0.95. Whether the
  training frames came from that camera is recorded nowhere: §16.1's
  "1024²" came from an illustrative shape in `nikon-control`'s docs.
- `fasterrcnn_resnet50_fpn(..., min_size=n, max_size=m)` loads the
  checkpoint with `strict=True` (the resize has no weights), resizes a
  1412² frame to `n`, and `transform.postprocess` returns the boxes in the
  frame's own pixels: a box at `(w/2, h/4, w, h/2)` of the resized image came
  back as `(706, 353, 1412, 706)` at 600, 800, 1100 and 1412.
- Warm time per 1412² frame, after one cold call: 0.23 s at 600 px,
  0.43 s at 800 (the default), 0.58 s at 1100, 0.96 s at 1412. At 1100 and
  1412 a synthetic frame hit the limit of 100 boxes above a score of 0.05.
- `normalize_plane` of a 12-bit frame and of the same frame times 16 differ
  by at most 6e-8, and an offset of 100 counts changes nothing: **the bit
  depth is not a parameter**. The same frame cut to 8 bits differs by 0.0027
  (quantisation); no smc frame is 8 bits.
- `nikon-control`'s review page (`nikon-control-dashboard`, page `/review`)
  is its only annotation tool that opens TIFFs; the others read ND2 only.
  It opens a folder of `images/<file_name>` and `annotations/val.json` (or
  `train.json`), COCO: `categories` `{id, name}`, `images` `{id,
  file_name, height, width}` and a per-image `reviewed` flag it sets,
  `annotations` `{id, image_id, category_id, bbox [x, y, w, h], area}` with
  `auto` (a proposal not yet confirmed) and `score`. Boxes are continuous
  edges in `[0, W]` (`train_detector.py`), the convention the model's output
  is in. Its `agreement.match_boxes` matches one-to-one, greedily by IoU
  ≥ 0.5; it has no per-class precision or recall.

**Pre-processing** — `smc/plugins/cells/detector.py`:

```python
@dataclass(frozen=True, slots=True)
class Preprocessing:
    percentiles: tuple[float, float] = (0.5, 99.5)
    input_size_px: tuple[int, int] | None = None
```

```python
MIN_INPUT_PX = 256
MAX_INPUT_PX = 4096


def model_input_size_px(
    image_shape_px: tuple[int, int],
    frame_um_per_px: float,
    model_um_per_px: float,
) -> tuple[int, int]: ...
```

- `percentiles` go to `normalize_plane`; `input_size_px` is torchvision's
  `(min_size, max_size)`, passed to `fasterrcnn_resnet50_fpn` when it is
  not `None` (the model's own 800 and 1333 otherwise).
  `TorchDetector(weights, *, device, score_threshold, preprocessing=Preprocessing())`
  and `load_detector(model_path, *, device, score_threshold,
  preprocessing=Preprocessing())` take it; the default is today's behaviour.
- `model_input_size_px`: `scale = frame_um_per_px / model_um_per_px`;
  `(round(min(h, w) × scale), ceil(max(h, w) × scale))`, so that the shorter
  side sets the scale and the longer one does not bind. A first value under
  `MIN_INPUT_PX` or a second over `MAX_INPUT_PX` is a `ValueError` naming
  both pixel sizes and the size it would give; a pixel size that is not
  finite and positive is a `ValueError`. A 1412² frame at 0.2756 µm shown to
  the model at 0.49 µm is `(794, 795)`.
- No crop and no tiling: the resize sets the pixel size the model sees, and a
  crop does not change it. Tiling (a frame larger than the model's input,
  cut and its boxes merged) is a follow-up if the measurement asks for it.

**Parameters** — `CellsParams` gains, after `merge_radius_um`:

```python
class CellsParams(PluginParams):
    model_um_per_px: float | None = Field(None, gt=0.0, le=10.0)
    low_percentile: float = Field(0.5, ge=0.0, lt=50.0)
    high_percentile: float = Field(99.5, gt=50.0, le=100.0)
    checked_um_per_px: float | None = Field(None, gt=0.0, le=10.0)
```

- `model_um_per_px`: the pixel size the model sees, in µm; `None` keeps the
  model's own resize. The plugin computes
  `model_input_size_px(scan.image_shape_px, scan.pixel_size_um,
  params.model_um_per_px)` **before** `load_detector` (a refused size does
  not wait for 166 MB), and its `ValueError` is a `PluginRunError` that
  names `model_um_per_px`.
- `low_percentile`, `high_percentile`: the intensity scaling, as trained.
- `checked_um_per_px`: the frames' pixel size at which this model and
  these parameters were measured (#342's script). It changes nothing in the
  detection; it is what `assumed` compares the scan with.
- With `detector=blobs`, the validator also refuses `model_um_per_px` or
  `checked_um_per_px` that is not `None`, and percentiles other than 0.5 and
  99.5 (`<name> is for detector=model: the blob detector ...`), as it
  refuses `model_path`.
- `assumed`, with `detector=model`, starts with one line when the scan was
  not measured: `checked_um_per_px` `None` gives `the model's detections
  were not measured on frames of this stand: checked_um_per_px is not set
  (docs/plugins/cells.md)`; a scan whose `pixel_size_um` is more than 5 %
  from it (`abs(p / c − 1) > 0.05`) gives `the frames are at {p:.4g} µm per
  pixel and the model was checked at {c:.4g} µm per pixel: its counts at this
  pixel size are not measured`. Within 5 %, no line. The merge radius line
  follows, as before.
- `CellsResult` does not change: the parameters are in the stage's
  `params`, and `model`, `model_sha256` already name the detector.
- Resume: the runner refuses a resume with other parameters, so a change of
  pre-processing is never mixed into one result.

**The measurement** — `smc/plugins/cells/evaluate.py`, numpy only:

```python
NO_BOX = "none"
DEFAULT_MATCH_IOU = 0.5


@dataclass(frozen=True, slots=True)
class LabelledBox:
    box_px: tuple[float, float, float, float]
    label: str
```

```python
@dataclass(frozen=True, slots=True)
class ClassScore:
    reference: int
    detected: int
    matched: int
    precision: float | None
    recall: float | None


def match_boxes(
    reference: Sequence[tuple[float, float, float, float]],
    detected: Sequence[tuple[float, float, float, float]],
    *,
    min_iou: float = DEFAULT_MATCH_IOU,
) -> list[tuple[int, int]]: ...
def confusion(
    reference: Sequence[LabelledBox],
    detected: Sequence[LabelledBox],
    *,
    min_iou: float = DEFAULT_MATCH_IOU,
) -> Counter[tuple[str, str]]: ...
def class_scores(
    counts: Counter[tuple[str, str]], classes: Sequence[str]
) -> tuple[dict[str, ClassScore], ClassScore]: ...
```

- Boxes are `(row0, col0, row1, col1)`, as `Detection`; IoU is
  `overlap.bbox_iou`.
- `match_boxes` is `nikon-control`'s rule: every pair with IoU ≥ `min_iou`,
  taken by decreasing IoU, then reference index, then detection index; a
  pair is kept when neither box is taken. One-to-one, labels ignored: a
  single found as a doublet is a match with the wrong label, not a miss and
  a false box. `min_iou` outside `(0, 1]` is a `ValueError`.
- `confusion` counts `(reference label, detected label)` over the matches,
  `(label, NO_BOX)` for a reference box not matched and `(NO_BOX, label)`
  for a detection not matched. Counters of frames add up.
- `class_scores`: per class `c` of `classes`, `reference` the pairs
  `(c, ·)`, `detected` the pairs `(·, c)`, `matched` `(c, c)`; `precision =
  matched / detected` and `recall = matched / reference`, **`None`** when the
  denominator is 0 (a class the detector never gives, as `debris` for
  `blobs`, has no precision: not 0, not nan). The second value is the same
  for any box, labels ignored: `matched` counts every pair with neither side
  `NO_BOX`.

**The script** — `scripts/dev/cells_check.py`, argparse, two commands. It
is run with an interpreter that has smc and torch (the stand's PC with the
`detect` extra; on a Mac, `../nikon-control/.venv/bin/python` with
`PYTHONPATH=src`), prints ASCII only, reads and writes text as UTF-8.

```text
cells_check.py export RUN_JSON OUT_DIR [--wells A1,B2] [--count 20]
    [--prefill model|blobs|none] [--prefill-score 0.3]
    [--model-path P] [--device auto|cpu|cuda|mps] [--model-um-per-px X]
cells_check.py measure DATASET_DIR [--detector model|blobs ...]
    [--model-um-per-px native,0.49,...] [--percentiles 0.5:99.5,...]
    [--score-threshold 0.5] [--max-overlap 0.7] [--match-iou 0.5]
    [--model-path P] [--device ...] [--json OUT]
```

- `export` reads the run document's `scan` result (`RunDocument.load`,
  `provider("scan")`, `ScanResult`), the run folder being the document's
  folder, takes the frames of `--wells` (all by default) in the scan's
  order, and keeps `--count` of them evenly spaced
  (`round(linspace(0, n − 1, count))`, all when fewer). It refuses an
  `OUT_DIR` that exists and is not empty: the owner's annotations are never
  overwritten. It writes:
  - `images/<basename of frame.file>`: the frame's file, copied unchanged
    (a duplicate basename is refused);
  - `annotations/val.json`: COCO as above, categories `single` 1, `doublet`
    2, `debris` 3, each image `reviewed: false`, and the prefill's
    detections (after `suppress_in_frame` at 0.7, at or above
    `--prefill-score`) as annotations with `auto: true` and their `score`.
    `x = col0`, `y = row0`, `w = col1 − col0`, `h = row1 − row0`;
  - `manifest.json`: the run document's file name (never its path, FM-50),
    the scan's `objective`, `pixel_size_um`, `image_shape_px`,
    `exposure_ms`, the prefill (detector, model file name and SHA-256, score,
    `model_um_per_px`), per image its `frame.file`, `well`, `x_um`, `y_um`,
    and the time of the export.
  `--prefill model` is the default: the owner deletes and corrects, which
  is faster than drawing every cell, and the reference leans towards the
  model's boxes, which `docs/plugins/cells.md` says.
- `measure` reads `annotations/val.json` and `train.json` (those that
  exist), keeps the images whose `reviewed` is true (none is an error: "mark
  the frames you checked as reviewed in /review and save"), and takes
  every annotation of those images as the reference, `auto` or not (a
  reviewed frame is the owner's verdict). The frames' pixel size is
  `manifest.json`'s. One configuration per `blobs` and per element of the
  product `model_um_per_px × percentiles` (`native` is `None`). Each runs
  the detector as `cells` does (`detect`, then `suppress_in_frame` at
  `--max-overlap`), sums the confusions, and prints per configuration the
  frames, the first frame's time and the mean of the others (cold and warm
  apart), a table per class (`reference`, `detected`, `matched`,
  `precision`, `recall`, `-` for `None`, three decimals), the line for any
  box, and the confusion's non-zero counts. `--json` writes the same
  numbers.

**Tests**: `tests/unit/test_cells_evaluate.py` (pure); the script in
`tests/test_dev_cells_check.py` with `blobs` on `PlateSample` frames (no
torch); the parameters in `tests/test_cells_demo.py` with the stub
detector; `input_size_px` reaching the model in
`tests/unit/test_cells_detector.py` (skipped without torch, run from the
torch venv).

## 10. Density (#193)

`smc/plugins/density/algorithm.py` and `plugin.py`, entry point
`density`. New, pure.

```python
class DensityParams(PluginParams):
    area_grid_um: float = Field(5.0, ge=0.5, le=100.0)
```

`requires = ()`, `reads = ("scan-plan", "scan", "cells")`,
`reads_optional = ()`.

```python
class GridTooFineError(ValueError): ...


def scanned_area_mm2(
    footprints: Sequence[Sequence[tuple[float, float]]],
    well: WellGeometry,
    *,
    grid_um: float,
) -> float: ...
def density(
    cells: Sequence[Cell],
    *,
    wells: Sequence[WellGeometry],
    areas_mm2: Mapping[str, float],
    fields: Mapping[str, int],
    failed_fields: Mapping[str, int],
    labels: Sequence[str],
) -> tuple[DensityWell, ...]: ...
```

- The area is the union of the footprints of the well's **taken** frames
  (failed fields excluded), intersected with the well shape, rasterised
  on a grid of `grid_um`. How it is rasterised, bounded and tested is
  refined below.
- A cell is counted when `well.contains(x, y)`; one outside its well's
  shape (a field corner on the plastic) counts in `outside_well`. Every
  label of `CellsResult.classes` appears in `counts`, zero included;
  `per_mm2` is `None` for an area of 0.
- `density` returns one `DensityWell` per entry of `wells`, in that order;
  `fields` and `failed_fields` (by well name) are what `DensityWell` of the
  same names holds, which the first version of this signature left no way
  to pass.

**Refined while planning #193** (measured 2026-10-28 on a prototype that is
not committed, numpy 2.5.3 on Python 3.13, numpy 1.26.0 and 2.2.6 on Python
3.10, `-W error::RuntimeWarning`; the same areas on the three):

- **The raster.** A footprint is a parallelogram given by its corners 0, 1
  and 3 (`geometry.footprint`'s order; corner 2 is checked to be `c1 + c3 −
  c0` within 1e-6 µm and a footprint of no area, a corner that is not
  finite or any other shape is a `ValueError` naming it). The grid covers
  the footprints' bounding box **cut to the well's circumscribed square**
  (the half-side is `size_um[0] / 2` for a disc and `hypot(size_um[0] / 2,
  size_um[1] / 2)` for a rectangle, around `center_um`: an extent, not a
  membership test), so a frame at a wrong stage position far from the well
  cannot make the grid huge. Points are the cell centres
  `(x_lo + (i + 0.5) · g, y_lo + (j + 0.5) · g)`, `g = grid_um`; a point
  counts when it is inside at least one footprint (`0 ≤ s, t ≤ 1` for
  `p = c0 + s · (c1 − c0) + t · (c3 − c0)`, solved with the entries written
  out) **and** `well.contains_many` says it is in the well. The area is
  the count times `g²`, in mm². Each footprint is tested over its own
  window of the grid, widened by one index each way (a rounding at the
  window's edge cannot lose a point), in bands of at most 2²⁰ elements.
  60 random cases (rotated, mirrored, non-square frames, round and
  rectangular wells turned by the plate) equal a brute force of every point
  against every footprint, and the area does not change with the band size.
- **What it measures.** A 96-well disc (305 fields of 388 µm) at 5 µm:
  32.1703 mm² for 32.1699, +0.001 %, in 0.03 s; at 0.5 µm 0.5 s and a peak
  of 41 MB above the baseline. A 6-well well (34.8 mm, 8013 fields) at 2 µm:
  1.1 s. A single field has the error of its edges, up to `g / (2 w)` on each
  axis (+1.3 % for a 333 µm field at 5 µm, measured), which is why the claim
  that the error is "under 1 % of a field" is dropped: it holds for the
  area of a well, not of one field. `assumed` says an edge is known to
  within half a step.
- **Bounds** (FM-74), applied before anything is allocated and on Python
  numbers: `grid_um` finite and positive; each axis at most 2¹⁷ points and
  the grid at most 2³⁰ points, tested on the **floats** first, written `not
  x <= bound` so that `nan` is refused too (`ceil(inf)` is an
  `OverflowError`: a step of 5e-324 µm raised it in the first prototype);
  the sum over footprints of their windows at most 2³¹ tests (measured 440
  to 650 million a second on the three numpys, so 3 to 5 s at the bound). The band is then at most `max(2²⁰, nx)` ≤ 2²⁰
  elements (a band of 2²⁴ peaked at 689 MB, one of 2²⁰ at 95 MB, the same
  time). A refusal is a `GridTooFineError`, whose message gives the sizes
  formatted with `:.3g` (a step of 1e-300 would print 300 digits otherwise);
  the plugin turns it into a `PluginRunError` that names the well and says
  to raise `area_grid_um`. A corner beyond 1e12 µm (1000 km of stage) is a
  `ValueError`: a corrupt record, and below it the arithmetic cannot overflow;
  a footprint farther than that from the well but within it is ignored (the
  extent is cut to the well), and the window indices are clipped on the
  floats before they become integers (FM-75). With one extra frame at 1e9,
  1e11 or 9.9e11 µm the 96-well area above is the same 32.1703 mm², with no
  numpy warning on the three numpys.
- **No matrix product** on coordinate arrays (FM-131), and a module-level
  `filterwarnings = error::RuntimeWarning` in the tests of the numeric
  module.
- **`density`** raises `ValueError` for: no label or a repeated one; a well
  listed twice; a well that `areas_mm2`, `fields` or `failed_fields` lacks;
  an area that is not finite and at least 0, or a count below 0; a cell of
  a well not in `wells`; a cell whose label is not in `labels`. So `counts`
  and `outside_well` of a well always add up to its cells. The cells of a
  well are tested with one `contains_many` call.
- **The plugin** reports a well when the scan took a frame in it **or**
  failed a field in it (a well whose every field failed is reported with 0
  fields, its failed count, an area of 0 and `per_mm2 = None`), in the scan
  plan's order; a well that the scan did not attempt is not reported, as the
  route leaves it out. The footprints are those of the scan's frames at
  their **readback** position, with the **scan's** matrix and frame shape
  (not the plan's: FM-124). It refuses, each with the stage's own words: a
  scan with no frame; a matrix that cannot be inverted (FM-125); cells
  counted on another number of frames than the scan holds, or on other wells
  (FM-111's case: the scan was run again); a cells file that no longer
  matches its result; a well that the plan does not list or cannot place.
  `labels` of the result are the cells' `classes`. `assumed` says that the
  area is rasterised on `area_grid_um` and an edge known to half a step, and
  that the wells' shape and position are the plate registration's and the
  plate's nominal sizes (the wall was not measured).

**Tests**: `test_a_full_disc_scan_measures_the_well_area` (within 1 % of
π r²); `test_overlapping_fields_are_not_counted_twice_in_the_area`;
`test_failed_fields_reduce_the_area`; `test_a_cell_on_the_plastic_counts_outside`;
`test_a_well_with_no_frame_has_unknown_density`;
`test_a_grid_too_fine_for_the_well_is_refused_before_allocating` (FM-74);
the plan of #193 lists the rest. **Failure modes**: FM-73, FM-74, FM-98,
FM-111, FM-124, FM-125, FM-131, FM-132.

## 11. Route (#194)

`smc/plugins/route/algorithm.py` and `plugin.py`, entry point `route`.
New, pure.

```python
class RouteParams(PluginParams):
    labels: tuple[str, ...] = ("single",)
    fields_per_well: int = Field(10, ge=1, le=500)
    min_cells: int = Field(1, ge=1)
    exclude_centre_um: float = Field(0.0, ge=0.0)
    wall_margin_um: float = Field(0.0, ge=0.0)
    objective: str | None = None
    candidate_step_fraction: float = Field(0.25, gt=0.0, le=1.0)
```

`requires = ()`, `reads = ("scan-plan", "scan", "cells")`,
`reads_optional = ("timings",)`. `exclude_centre_um` is the **radius** of
the central disc excluded; `labels` must be among the cells' `classes`.

**The imaging objective**: `None` is the scan's, with the scan's matrix.
Another one needs a turret (`microscope.get(ObjectiveTurret)`) and its
profile size: the scan's matrix scaled by `size(objective) / size(scan
objective)`, `assumed` gaining `the field of <objective> is the scan's
scaled by the profile's pixel sizes`; no turret or no size is a
`PluginRunError`.

```python
@dataclass(frozen=True, slots=True)
class ChosenField:
    x_um: float
    y_um: float
    counts: Mapping[str, int]


def allowed(
    x_um: float,
    y_um: float,
    well: WellGeometry,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    exclude_centre_um: float,
    wall_margin_um: float,
) -> bool: ...
def choose_fields(
    cells: Sequence[Cell],
    well: WellGeometry,
    *,
    labels: Sequence[str],
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
    n: int,
    min_cells: int,
    exclude_centre_um: float,
    wall_margin_um: float,
    step_um: float,
) -> tuple[ChosenField, ...]: ...
def order_route(points: Sequence[tuple[float, float]], *, start: int) -> list[int]: ...
def path_length_um(
    points: Sequence[tuple[float, float]], order: Sequence[int]
) -> float: ...
```

- **`allowed`**: every corner of the footprint is inside the well shrunk
  by `wall_margin_um` (`contains`), and, when `exclude_centre_um > 0`, the
  footprint polygon stays at least `exclude_centre_um` from the well's
  centre (the polygon's distance to the point, zero when the point is
  inside).
- **`choose_fields`** (greedy, deterministic): candidate centres on a
  square lattice of `step_um` (`candidate_step_fraction × min(fov)`) over
  the well's bounding box, kept when `allowed`, numbered row by row from
  +Y down, then +X. A cell is in a field when `pixel_of_stage` falls
  inside the frame. Each round picks the candidate holding the most
  wanted cells not yet held (ties: the lowest number), stops when that is
  below `min_cells` or `n` fields are chosen, and removes the candidates
  whose footprint would overlap the chosen one (same orientation:
  `|d_col| < W` and `|d_row| < H` in pixels), so no cell is imaged twice.
  Counting is numpy in chunks (FM-74); what is counted is refined below.
- **`order_route`**: nearest neighbour from `start`, then 2-opt on the
  open path with the first point fixed, Euclidean, until no swap shortens
  it by more than 1e-9 µm. The cost is distance: `timing.py`'s model is
  affine in distance (an overhead per move plus a rate), so the shortest
  path is also the fastest under it. Whether the Ti2's stage is closer to
  Chebyshev (both axes at once) is not measured (#198).
- **Across wells**: wells in the scan plan's order; the first well's
  route starts at its field nearest the first planned field of that well,
  every later one at its field nearest the previous well's last position.
- Each position takes `z_um` and `pfs_offset` from the nearest taken
  frame of its well that is not `unlocked` (decided for #232, below);
  `name` is `<well>_r<k:03d>`; `counts` holds every
  label inside the footprint. `travel_s` is the sum of `timings.move_ms`
  over the route when `timings` is usable, else `None`.

**Refined while planning #194** (measured 2026-10-28 on a prototype that
is not committed, with numpy 2.5.3 on Python 3.13 and numpy 1.26.0 and
2.2.6 on Python 3.10):

- **Counting is not cells × candidates.** Per cell, only the lattice
  points inside the footprint's bounding box around it can hold it: a
  window of `floor(W_um / step) + 3` by `floor(H_um / step) + 3` points
  (`W_um`, `H_um` are `fov_um`), each then tested exactly. It gave the
  counts of the brute force on 283 random cases (rotated, mirrored,
  non-square, round and square wells). For 300 000 cells in a 96-well
  well at 40× it took 0.6 s where cells × candidates took 12 s, and 0.6 s
  again in a 6-well well, whose lattice has 34 times more candidates.
  Bounds, checked before anything is allocated and on Python numbers, not
  numpy integers (FM-74): the lattice holds at most 250 000 points (a step
  of 5e-324 µm makes `int(half / step)` an `OverflowError`, and
  `candidate_step_fraction` accepts 1e-9); cells × window at most 2³¹
  tests; a chunk of at most 2¹⁸ elements (about 94 bytes each at the peak:
  25 MB). A refusal is a `ValueError`, which the plugin turns into a
  `PluginRunError` that names the field's size.
- **No matrix product.** The pixel offset is computed from the four
  entries of the inverse, written out: numpy 2.2.6 (Python 3.10, macOS
  arm64) raised `RuntimeWarning: divide by zero encountered in matmul` for
  `@` of a finite (N, 2) array by a 2 × 2 matrix from N = 2000 up (not for
  1500, and not with numpy 1.26.0 or 2.5.3), which `-W error` turns into a
  failure. The inverse comes from `pixel_of_stage` at three points, so the
  sign convention stays in `geometry.py` and a singular matrix is refused
  as it is there.
- **The lattice** is centred on the well's centre (the centre is a
  candidate and the lattice is symmetric) and reaches `floor(half /
  step)` steps each way, `half` being the half-extent of the well's shape
  in the stage axes (a square well turned by the plate: that of the
  rotated rectangle). Numbering is row by row from +Y down, then +X.
- **Membership and overlap.** A cell is in a field when its pixel offset
  from the field's centre is within `W / 2` and `H / 2`, edges included.
  Two fields overlap when `|d_col| < W − 1e-6` and `|d_row| < H − 1e-6`:
  fields that touch are allowed. Each candidate is counted once, and a
  count only goes down, so the greedy is lazy: the best candidate is
  counted again over the cells that no chosen field holds, and when its
  count fell (a cell exactly on the edge that two touching fields share
  was held by the first) the new count replaces the old one and the round
  starts again. That is the exact greedy (equal to recounting every
  candidate every round on 150 inputs with cells on the lattice's edges),
  it never chooses a field with fewer than `min_cells` new cells, and it
  counts a cell once; `counts` and `uncovered` follow from it.
- **Another objective.** The scan's own objective is not "another one": it
  needs no turret and gains no line. The scan's `z_um` and `pfs_offset`
  belong to its objective (FM-118: an autofocus offset is one recorded for
  a sample and an objective), so for another objective every position's
  `z_um` and `pfs_offset` are `None`, and `assumed` gains `z_um and
  pfs_offset are unknown for <objective>: the scan's were measured with
  <scan objective>` and `no offset between the two objectives is applied`.
- **Reported.** `assumed` always says that the route minimises the
  Euclidean distance between fields and that the stage's move time against
  it is not measured. `uncovered` has every scanned well, in plan order,
  zeros included; `counts` of a position has every wanted label, zeros
  included. `distance_um` and `travel_s` cover the legs between wells too:
  the first position has no leg. `name`'s `k` counts from 0 along the
  well's route. `fov_um` is the extent of the (scaled) footprint.
- **No route is no result.** When no well has a chosen field the stage is
  a `PluginRunError` that says which of two things happened: the cells
  stage found none of the wanted labels, or it found some and no allowed
  field holds `min_cells` of them. A route of no position is of no use to
  the export, and `done` would say that it is.

**Decided for #232** (the owner's call, 2026-10-29; found by the adversarial
review of #230): a stop's `z_um` and `pfs_offset` do not come from a frame
whose autofocus had lost the sample.

- **The rule.** A scan frame's `focus` is `locked`, `unlocked`, `off` or
  `none` (§3.4). The *candidates* of a well are its frames whose `focus` is
  not `unlocked`. A stop takes **both** values from one frame, the nearest
  candidate by the frame's readback (ties: the lowest index in
  `scan.frames`, as before), and both are `None` when the well has no
  candidate: no frame at all, or only `unlocked` ones. The two values never
  come from different frames.
- **Why `off` and `none` count.** `off` is an autofocus the scan did not
  touch, and `none` a stand without one: their `z_um` is the focus stage's
  reading under the operator's focus, which is what the scan measured, and
  their `pfs_offset` is `None`. Only `unlocked` says the reading was taken
  while the autofocus was on and searching, so its `z_um` is wherever the
  search had got to and its `pfs_offset` is the stored one that nobody read
  locked on this sample (FM-118): the export (§14) would hand it on as the
  operator's. Unknown beats guessed.
- **Reported.** When the rule changed the answer of at least one stop, that
  is when the nearest frame of **all** the well's frames was `unlocked`,
  `assumed` gains one line, after the lines above and with `n` the stops
  concerned, `total` all stops and `m` those of the `n` that have no
  candidate: `{n} of {total} stops had an unlocked frame as their nearest
  frame (the autofocus had not locked on it): {n - m} took z_um and
  pfs_offset from the nearest frame that was not unlocked, {m} have none
  because every frame of their well was unlocked`. A stop whose nearest
  frame is not `unlocked` is not counted, whether or not a farther
  `unlocked` frame exists; a scan without `unlocked` frames adds no line;
  for another objective every stop has `None` already and the line is not
  added.
- **`fov_of`** is one function, `smc.plugins.geometry.fov_of` (§3.1);
  `route.algorithm` and `scan_plan.algorithm` import it and no longer define
  it.
- **Not changed.** A stop is looked at between two wells and not inside
  one: the check inside `choose_fields` would need a callable in its
  signature for a well that takes 0.3 to 0.45 s at the default step, and
  17 s only at the bound (760 000 cells and a step of 0.02 of the field,
  both the operator's choice); Ctrl-C ends it at once. Reopen it if the
  plate page's Stop shows the wait at a stand.

**Tests** (hypothesis where marked): `test_no_field_leaves_its_well_or_enters_an_exclusion`
(hypothesis over margins, exclusions, rotations); `test_chosen_fields_never_overlap`;
`test_the_greedy_takes_the_densest_field_first`;
`test_the_route_visits_every_field_once` (hypothesis);
`test_two_opt_never_lengthens_the_route` (hypothesis, against the
nearest-neighbour path); `test_wells_follow_the_scan_order`;
`test_another_objective_scales_the_field`; `test_travel_time_is_none_without_timings`;
the plan of #194 lists the rest. **Failure modes**: FM-73, FM-74, FM-98,
FM-104, FM-118, FM-121, FM-124.

## 12. Timing (#24)

`smc/plugins/timing/algorithm.py` and `plugin.py`, entry point
`timing = "smc.plugins.timing.plugin:TimingPlugin"`, slot `timings`. A port
of `timing.py` with its 19 tests; `Budget` and `plan()` stay in
`algorithm.py`. `TimingsResult` (§3.5, `slots/timings.py`) is merged and does
not change: `scan-plan` and `route` already read it.

```python
class TimingParams(PluginParams):
    repeats: int = Field(3, ge=1, le=20)
    distances_um: tuple[float, float] = (200.0, 9000.0)
    with_autofocus: bool = True
    channels: tuple[str, ...] = Field((), max_length=32)
```

`requires = (XYStage, Camera)`, `reads = ()`, `reads_optional = ()`,
`provides = "timings"`. `distances_um` is `(short, long)`: each in `(0,
50 000]` and strictly increasing, since two equal distances fit no line
(`fit_line` would return their mean as an overhead and a rate of 0, with no
warning). `channels` holds no empty name and no repeat (a repeat would
overwrite its own entry in `channel_ms`).

**Refined while planning #24** (measured 2026-10-29 on the demo and on
`FakeCore.demo_like()`, Python 3.13.0, pymmcore-plus as in the venv):

- **What the demo costs.** A 200 µm move reads 25 ms, 9000 µm 904 ms (the
  demo stage moves at 10 mm/s: the fit is an overhead of about 5 ms and
  100 ms/mm). A snap reads 11 ms at a 10 ms exposure. `engage()` returns
  `True` in 0.1 ms, from off and while on. The demo has `Autofocus` and
  `Channels` (`Cy5`, `DAPI`, `FITC`, `Rhodamine`; current `DAPI`); a switch
  reads 0.1 to 0.3 ms. `limits_um()` is `None`. The defaults take 5.5 s of
  stage time on the demo.
- **Stage.** `xy.wait()`, then `start = xy.position_um()` (FM-85); a start that
  is not finite refuses the run before anything moves (FM-125). For each
  distance, `repeats` times: `ctx.raise_if_stopped()`, the timed
  `move_to_um(start.x_um + s·d, start.y_um)`, `ctx.raise_if_stopped()`, the
  timed `move_to_um(start.x_um, start.y_um)`. `s` is `+1`, or `-1` when the
  soft limits (`limits_um()`) put `start.x_um + d_long` outside the X travel and
  `start.x_um - d_long` inside it; when neither fits, the run is refused before
  anything moves, with the limits, the start and the distance in the text. With
  limits `None` the moves go `+X` and `assumed` says `the stage reports no soft
  limits: the probe moves were not checked against its travel`. The timed call
  is the facade's `move_to_um`, which returns once the stage has arrived: the
  cost the scan pays. The model's distance of a point is the **median of the
  distances the stage read back** (FM-142), not the commanded one, and its time
  is the median of its `2·repeats` samples. A move whose readback distance
  differs from the one it was asked by more than `max(5 µm, 2 %)`, or is not
  finite, is not a measurement of that distance: the loop completes (the stage
  is back at the start), then the run fails with a `PluginRunError` that says
  how far the stage went. The line fit is `fit_line`, clamped at 0, as in the
  port.
- **Autofocus.** Read **before any move**: `Autofocus` present, `with_autofocus`,
  `is_engaged()` and `is_locked()`. The `is_locked()` read is what makes the
  stand's offset a *seen* offset on a stand that declares
  `engage_drives_z_to_offset` (m2 §12 D1; a `disengage()` does not forget it),
  so the engage below is allowed there. Measured iff all four hold. Then,
  `repeats` times: `ctx.raise_if_stopped()`; `is_engaged()`: if it is no longer
  on, something else switched it off (the operator, or a lost lock that the
  protocol answered), so the loop **does not switch it on** (FM-129) and stops;
  `disengage()`, then the timed `engage()`. The first sample that does not lock
  (`engage()` returns `False`) or whose `engage()` is refused
  (`SafetyRefusedError`: nothing was sent) **ends the section**: another sample
  would wait the whole lock timeout again, and a search drives Z. `refocus_ms`
  is then `ASSUMED_REFOCUS_MS` (400) and `assumed` says why, how many samples
  locked and their median (the median may exceed 400). Every other error
  propagates, a halt and an excursion among them (FM-64): the port's `except
  Exception` took them for "did not lock". Autofocus present but not measured
  (`with_autofocus` false, off, or on but not locked): `refocus_ms = 400`,
  assumed, saying which. Autofocus absent: `refocus_ms = None`. After the
  section, if the plugin switched the autofocus off and it did not lock again,
  it stays off and `assumed` says `the autofocus was switched off for the
  measurement and did not lock again; engage it again before the scan`; the
  plugin never engages it again after a non-lock (FM-88).
- **Snap and channels.** `snap_ms` is the median of `repeats` bare snaps, at the
  channel and exposure the stand has. With `channels`: the stand needs
  `Channels` and every name must be one of `names()`, else the run is refused
  before anything moves or switches. The channel found is `current()`; the
  names are visited in the order given, except the one equal to the channel
  found, which goes **last**, so that every timed switch changes channel (a
  switch to the channel already selected can cost nothing and would pull the
  median down). Per name: `ctx.raise_if_stopped()`, the timed `select(name)` (one
  switch sample), then `repeats` timed snaps whose median is `channel_ms[name]`.
  When the only name is the channel found, nothing is selected,
  `channel_switch_ms` is `None`, and `assumed` says no switch was timed.
  `channel_switch_ms` is the median of the switch samples, or `None` with no
  `channels`. A `current()` of `None` (the stand matches no channel) cannot be
  restored: a WARNING says the stand is left on the last channel, and the run
  goes on. After a **normal** end, a stand not on the channel found is selected
  back to it (a restore that raises propagates). `samples` counts the timed
  operations whose median is in the result: `4·repeats` (stage), `repeats`
  (autofocus, when measured), `repeats` (bare snaps), and per channel `repeats`
  snaps plus the switch samples.
- **Not normal.** After a stop, a halt or a fault nothing is restored and
  nothing is caught (FM-88, FM-64): a WARNING per thing left, each read in its
  own `try` (the stage between its two moves: where it is and where it started;
  the autofocus the plugin switched off; the channel it left the stand on). A
  stop is the `StopRequested` of `ctx.raise_if_stopped()`, so the record is
  `stopped` and no slot is filled.
- **Dry run** is refused with a `PluginRunError`: the calls that would be timed
  are skipped, and their durations are not the stand's. No checkpoint is kept:
  a stopped timing is measured again, and nothing half-done is stored (FM-106).
- **Clock.** `time.perf_counter` through a module attribute `_clock` that the
  helper `_time_ms(fn)` reads at call time, so a test replaces it
  (`time.monotonic` ticks every 15.6 ms on Windows).

```python
ASSUMED_CHANNEL_SWITCH_MS = 100.0
ASSUMED_REFOCUS_MS = 400.0


def fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float]: ...
def channel_cost_ms(timings: TimingsResult, channel: str) -> float: ...
def position_ms(
    timings: TimingsResult,
    channels: Sequence[str],
    move_um: float = 0.0,
    *,
    refocus: bool = True,
) -> float: ...
def describe_timings(timings: TimingsResult) -> list[str]: ...
def plan(
    timings: TimingsResult,
    channels: Sequence[str],
    interval_s: float,
    *,
    requested: int = 0,
    move_um: float = 2000.0,
    refocus: bool = True,
    headroom: float = 0.8,
    duration_h: float = 0.0,
) -> Budget: ...
```

```python
@dataclass(frozen=True, slots=True)
class Budget:
    per_position_ms: float
    interval_s: float
    positions: int
    requested: int = 0
    duration_h: float = 0.0
    assumed: tuple[str, ...] = ()
```

`Budget` has the port's properties `timepoint_s`, `fits`, `timepoints`, `duty`
and `describe()`. `fit_line` is the port's `_fit_line`, public. The three
functions replace the methods of the port's `Timings`, which is now
`TimingsResult`. `channel_cost_ms` is the channel's measured cost, else the
mean of the measured channels, else `snap_ms` (the port returned 0, which
counted an exposure as free). `position_ms` is `move_ms(move_um)`, plus
`refocus_ms` when `refocus` and it is not `None`, plus, for each channel,
`channel_switch_ms` (`ASSUMED_CHANNEL_SWITCH_MS` when it is `None`) and its
cost; with no channel, one `snap_ms` and no switch. `plan` raises `ValueError`
for an `interval_s` that is not finite and positive, a `headroom` outside
`(0, 1]`, or a `requested`, `duration_h` or `move_um` that is not finite and at
least 0 (FM-32; the port clamped a headroom). `Budget.assumed` is the timings'
own `assumed`, plus a line for the switch cost that `plan` assumed and one for
each channel whose cost is a mean or a bare snap. A per-position cost of 0
gives `positions = 0`, as in the port. `describe_timings` is the port's
`Timings.describe`: the move model, the refocus (`not applicable: no
autofocus` for `None`), the switch (`not measured` for `None`), each channel,
the snap, and the assumptions; the plugin logs it at INFO (`smc run -v`).
There is no CLI command or page for `plan` in M3 (the scan takes one channel
per field and has no interval): a program calls it, and a surface for it is a
later issue.

`scan-plan`'s plugin carries the timings' `assumed` into the plan's own
`assumed` as `timings: <line>`, one per line, whenever it computes an
estimate: an estimate whose refocus was assumed must not read as measured.

**Tests** (the plan of #24 lists them): the port's twelve pure tests and its
seven loop tests (the demo ones on the facade, the `FakeCore` ones on
`fake_microscope`); a fake clock and a stage that advances it by a known cost
make the fit exact (FM-124); a stage that stops short (FM-142); a PFS that
never locks (FM-88, FM-118); a PFS switched off from outside (FM-129); a halt
during the engage (FM-64). **Failure modes**: FM-12, FM-19, FM-43, FM-44,
FM-64, FM-85, FM-88, FM-118, FM-124, FM-125, FM-129, FM-142.

## 13. The static report (#195)

`smc/plugins/report/svg.py`, `render.py` and `plugin.py`, entry point
`report`. `svg.py` holds the drawing primitives (numbers, escaping, colours,
paths, the subsample), `render.py` the page; both are pure and import
nothing of `smc.hardware`.

```python
class ReportParams(PluginParams):
    max_cells_drawn_per_well: int = Field(2000, ge=0, le=20_000)
```

`requires = ()`, `reads = ("scan-plan", "scan", "cells", "density")`,
`reads_optional = ("route",)`, `provides = "report"`.

```python
def render(
    *,
    plan: ScanPlanResult,
    scan: ScanResult,
    cells: Sequence[Cell],
    density: DensityResult,
    route: RouteResult | None,
    title: str,
    facts: Sequence[tuple[str, str]],
    assumed: Sequence[str],
    max_cells_drawn_per_well: int,
) -> str: ...
```

**One HTML file**, `report/<token>.html`, written UTF-8 through
`write_atomically`, with `<meta charset="utf-8">`, inline SVG and CSS,
**no script and no external URL** (it opens offline on the microscope PC
and attached to an issue). It shows:

1. A header: `title`, then a table of facts (the plate, the objective, the
   wells scanned, the fields taken and failed, the cells found, then the
   `facts` the plugin passes, §13 *Refined*), and every `assumed` line of the
   upstream results, labelled by stage by the plugin (`scan-plan: …`).
2. The plate map, one per label: each well drawn in stage µm (Y up), its
   fill the density per mm² on one sequential scale per label, the value
   written in it, a legend; unscanned wells outlined only.
3. The density table (well × label: count, per mm², area, failed
   fields).
4. Per scanned well: the frames' footprints, the cells as dots coloured by
   label (at most `max_cells_drawn_per_well`, a deterministic
   subsample, the caption saying how many are drawn), and, with a route,
   the chosen fields as rectangles with their order and the route's line.
5. With a route: the route across the plate, its length and `travel_s`.

Every text from a result goes through `html.escape`.

**Refined while planning #195** (measured 2026-10-28 on a prototype that is
not committed, Python 3.13, pymmcore-plus 0.18.1, useq-schema 0.9.2;
rendered by Google Chrome headless on macOS, not by Edge, Firefox or Safari):

- **The signature gained `facts`.** The header asks for the run document's
  file name, the profile and a date, which no argument of the first
  signature carried. `facts` is a sequence of `(name, text)` rows that the
  plugin builds and `render` prints after its own rows, escaped: `("Run
  document", ctx.run_doc.name)`, `("Profile", …)`, `("Generated", "2026-10-28
  14:05 UTC")`, `("Detector", "<weights file name> (sha256 <12 hex>), score ≥
  0.5, merge radius 5 µm, cpu")`. **"Generated", not an experiment date**: no
  record carries the date of the imaging, and the report does not invent one.
  The time lives in the plugin, so `render` is a pure function of its
  arguments (the same records give the same bytes). The file name needs the
  one **addition to §1.4**: `RunContext.run_doc: Path`, the path the runner
  was given (the context already held it, privately, for `save_checkpoint`); a
  plugin reads its `.name` and never writes it. Only the name goes into the
  report, never a path (FM-50: a report is attached to issues).
- **Size, measured.** 96 wells, 305 fields each (29 280 footprints), 2 000
  cells each: **4.7 MB** with one `<path>` per label and well (a dot is the
  subpath `M<x> <y>h0.1` stroked with a round cap, 16 bytes a cell), **12.2
  MB** with one `<circle>` per cell; at 5 000 cells 9.2 MB against 27.9 MB
  (and about 480 000 elements for a browser to lay out). The footprints of a
  well (305 parallelograms in one `<path>`) are 17 kB; building the whole
  string takes 0.1 to 0.3 s. So: one path per label and well, never an
  element per cell. A cap per well is no bound on the page (FM-74: `20 000`
  cells × 96 wells is 1.9 million dots), so `render` also holds a **total
  of 300 000 dots** (`MAX_CELLS_DRAWN_TOTAL`, 4.7 MB of dots): each scanned
  well draws at most `min(max_cells_drawn_per_well, MAX_CELLS_DRAWN_TOTAL //
  scanned_wells)`, and the caption of a well that was cut says `drawn of
  total`. The default is 2 000, not 5 000: at 320 px a well holds 2 000 dots
  at 18 % coverage and 5 000 at 44 %, where the dots stop telling a density
  from another. The footprints and the route's fields are not capped: each is
  a frame the scan took (a TIFF on disk) or a position of the route's own
  record.
- **The subsample is exact and deterministic**: with `n` cells to draw in a
  well and a cap `c < n`, the cells at indices `(i * n) // c` for `i` in
  `range(c)` (distinct, increasing, exactly `c`), in the order of the cells
  file, so frames are thinned evenly and the same records give the same
  page. `c = 0` draws none and says so.
- **Only the cells density counted are drawn**: those `WellGeometry.contains`
  puts inside the well (the cells that `outside_well` counts are not drawn,
  and the caption gives their number), so an uncapped panel holds exactly
  `DensityWell.counts[label]` dots per label.
- **Coordinates.** A well's panel has the well's centre at the origin and Y
  down: a point `(x, y)` of the stage is drawn at `(x − cx, cy − y)`, a
  number rounded to 0.1 µm (`-0` is written `0`, a value that is not finite
  is a `ValueError`, §13 *Refused*). The plate map and the route map have one
  origin for the whole plate, Y flipped the same way, so A1 is at the top
  left when the stage's Y of row A is larger than that of row B (measured on
  a plate turned by 0.8°). **No `transform` flips a group**, because text in
  it would be mirrored. A footprint is `footprint(x, y, matrix=…,
  image_shape_px=…)` of the **scan** (a frame is what the scan saw, FM-124),
  at the stage's readback (as the frame's own `well_dx_um`, `well_dy_um` say
  it, §8 *Refined while planning #430*, or `x − cx`, `y − cy` for a frame that
  has none), drawn as one closed polygon; a **failed** field,
  which has only its target, is drawn the same way, dashed; a chosen field of
  the route with the **route's** matrix and frame shape (another objective is
  another field). A square well is a polygon of its four corners turned by the
  plate's rotation, a round one a `<circle>`. A dot's diameter is
  `max(0.012 × the well's width, 12 µm)`.
- **Colours.** The label colours are the Okabe-Ito set by the label's index
  in `density.labels` (`#D55E00`, `#0072B2`, `#009E73`, `#CC79A7`, `#E69F00`,
  `#56B4E9`, `#7A7A7A`, cycling), so a legend and a dot agree. The scale of a
  label is white to its colour, linear from 0 to the largest density of that
  label among the scanned wells: `mix = 0.10 + 0.90 × t`, so that a well of
  density 0 is paler than any other but not white, and different from an
  unscanned well, which has no fill. The value is written in black or white,
  whichever has the larger WCAG contrast with the fill (black when the fill's
  relative luminance is above 0.179), so at least 4.58 : 1 on every step. The
  legend is five swatches (`t` = 0, 0.25, 0.5, 0.75, 1) with their values.
  No `linearGradient`, no `pattern`, no `url(#…)`: a fragment is the only kind
  of reference in the file. A value is written with no exponent (`1234`,
  `45.6`, `0.12`: whole numbers from 100, one decimal from 10, two below), and
  not at all on a plate of more than 384 wells (the table and the `<title>`
  of the well hold it).
- **Two unknowns are not one zero.** `per_mm2 is None` with `fields == 0` is
  written *no frame: unknown*; with `fields > 0` (a field smaller than the
  area grid's resolution: the sampled zero of FM-138) it is written *area
  below the grid's resolution (<g> µm)*; a measured 0 is `0`. The header
  states the area grid and that an edge is known to within half a step (from
  `density.assumed`).
- **What a reader can verify** (and the tests do, through
  `html.parser`, with no browser): the page is HTML that balances; every
  `<svg>` has `role="img"` and an `aria-label`; the data hooks are
  `data-` attributes and classes, fixed: `table.density` with one
  `tr[data-well]` per reported well; `svg.plate-map[data-label]` with one
  `g.well[data-well][data-state]` (`density`, `unknown` or `unscanned`) and
  `data-value` for a density; `section.well-detail[data-well]` with, in its
  `svg`: `path.frames[data-n]`, `path.failed[data-n]`,
  `path.cells[data-label][data-drawn][data-total]` and, with a route,
  `path.field[data-order][data-name]` and `polyline.route-well`; `svg.route-map`
  with `polyline.route[data-stops]`. A well's panel is linked from its map
  entry by `<a href="#w<k>">`, `k` the well's index in `density.wells`, never
  built from text.
- **Text is escaped by one function** (`html.escape`, `quote=True`), and
  nothing from a record is used as an `id`, a class, a style or a number
  without passing `num`. Free text from a record is: the plate and objective
  names, the labels, a route stop's `name`, every `assumed` line, `title` and
  `facts`. A well's name is not free (it must name a well of the plate, or
  `well_geometry` refuses it). The CSS is one static `<style>`, with
  `font-family: system-ui, "Segoe UI", sans-serif`.
- **Refused** (a `ValueError` of `render` that ends `; run <stage> again`,
  which the plugin turns into a `PluginRunError`): a well that is not on the
  plate (`scan-plan`); a coordinate or a size that is not finite, in the
  plate's registration, a frame, a cell or a stop (the stage that holds it;
  the plate's `useq` model accepts `nan`, FM-114).
  The **plugin** refuses, before it draws, a set of records that do not fit
  each other (FM-111): density's labels are not the cells' classes; a
  density well whose `fields` or `failed_fields` differ from the scan's, or
  whose `sum(counts) + outside_well` differs from the number of cells the
  file holds for it (run `density` again); a cell in a well density does not
  report; a route with a label the cells do not have or a stop in a well
  density does not report (run `route` again). A cells file that no longer
  matches its result is `load_cells`' `ValueError`: run `cells` again.
- **Written** as `<token>.html` under `report/`, a new token for each run; a
  write that fails says what is kept (`AtomicWriteError.kept`) and that
  `report` is to be run again, since nothing is resumed. A stop ends the stage
  before the write. A dry run writes the report (nothing in it needs the
  stand).

**Tests** (the first five were this section's first list; the fourth is
renamed because a well's name cannot carry markup):
`test_the_report_is_one_file_with_no_script_and_no_url`;
`test_every_scanned_well_appears_with_its_density`;
`test_text_with_markup_is_escaped_wherever_it_is_shown`;
`test_the_report_without_a_route_says_so`;
`test_the_cells_drawn_are_capped_and_the_caption_says_so`; and, from FM-124,
`test_footprints_use_the_scans_matrix_and_frame_not_the_plans`,
`test_the_cells_drawn_are_the_cells_density_counted`,
`test_route_fields_are_drawn_in_the_routes_frame`,
`test_the_total_of_cells_drawn_is_bounded_whatever_the_cap`,
`test_a_non_finite_coordinate_is_refused_naming_the_stage`; the plan of #195
lists the rest. **Failure modes**: FM-21, FM-24, FM-38, FM-50, FM-73, FM-74,
FM-98, FM-111, FM-114, FM-124, FM-125, FM-131, FM-138, and two new entries
(markup built from a result's text; an output bounded per unit but not in
all).

## 14. The NIS-Elements export (#196)

`smc/plugins/nis_export/format.py` and `plugin.py`, entry point
`nis-export`. The format is the one of a real XY position list (multipoint)
that NIS-Elements 6.20 exported on the Ti2-E on 2026-10-29 (#196); what
was measured and decided from it is under *Refined while planning #196*
below.

```python
SUFFIX: str


@dataclass(frozen=True, slots=True)
class NisPosition:
    name: str
    x_um: float
    y_um: float
    z_um: float | None
    pfs_offset: float | None


def encode(positions: Sequence[NisPosition]) -> bytes: ...
def decode(data: bytes) -> tuple[NisPosition, ...]: ...
```

```python
class NisExportParams(PluginParams):
    out: Path | None = None
    overwrite: bool = False
```

`requires = ()`, `reads = ("route",)`, `provides = "nis-export"`. The
positions are the route's, in order. `out=None` writes
`nis-export/<token><SUFFIX>`; an `out` elsewhere (a folder NIS reads) is
written there, refused when it exists and `overwrite` is false, and stored
as its absolute POSIX path. `z_mode` says which of `z_um` and
`pfs_offset` the format carries. **Acceptance** is #196's: the
sample's structure round-trips (`decode(encode(x)) == x` and the sample
decodes), and the owner loads one export into NIS on the stand.

**Refined while planning #196** (measured 2026-10-29 on the sample,
`local/nikon-positions/multipoints.xml`, 8846 bytes, 10 points, with a
prototype that is not committed, on Python 3.13 and 3.10):

- **The file.** UTF-16 little-endian with a byte-order mark (`FF FE`), no
  line break anywhere, none at the end. The text is the declaration
  `<?xml version="1.0" encoding="UTF-16"?>`, then `<variant
  version="1.0"><no_name runtype="CLxListVariant">`, two flags
  `bIncludeZ` and `bPFSEnabled` (`runtype="bool"`, `value="true"` or
  `"false"`), one `Point00000`, `Point00001`, … per position (five digits,
  from 0, `runtype="NDSetupMultipointListItem"`), and
  `</no_name></variant>`. Every leaf is `<tag runtype="…" value="…"/>`,
  `runtype` first, no space before `/>`. A point holds, in this order:
  `bChecked` (bool), `strName` (`CLxStringW`), `dXPosition`,
  `dYPosition`, `dZPosition`, `dPFSOffset` (`double`), `baUserData`
  (`CLxByteArray`, `value=""`). A double is written with 15 decimals
  (`f"{v:.15f}"`: `-11613.300000000001091`).
- **What the sample says about units.** The positions are those of a 96-well
  plate in µm (X −13 000 to −4 000, Y 3 500 to 9 800, Z 4089.0 to 4089.12),
  on the same 0.1 µm steps as Micro-Manager's readings of this stage
  (`-28779.100000000002` in session 1). `dPFSOffset` is 9370 in every point
  although `bPFSEnabled` is false: NIS writes its current offset whatever
  the flag. 9370 is also the offset Micro-Manager's `PFSOffset` read in
  session 1 ("left from earlier use", FM-118), so the two share a unit.
  That NIS shares Micro-Manager's origin and direction of X, Y and Z is
  consistent with these numbers and **not measured**.
- **The encoder reproduces NIS's bytes.** With the sample's own values
  (offset 9370 included), the prototype's output equals the sample byte for
  byte except the `bPFSEnabled` flag it set; with the offset absent it
  differs only in the ten `dPFSOffset` values (below).
- **A flag carries the field.** `encode` sets `bIncludeZ` when every
  position has `z_um`, `bPFSEnabled` when every one has `pfs_offset`, and
  refuses a list where only some have one (`ValueError`): a list NIS reads
  with one flag cannot say that a value is unknown for one point.
  `decode` reads `dZPosition` as `z_um` only under `bIncludeZ`, and
  `dPFSOffset` as `pfs_offset` only under `bPFSEnabled`, else `None`. A
  field without its flag is written as `0.000000000000000`, a number that
  is plainly no reading; the sample's unused 9370 is not reproduced.
- **The plugin's three modes.** `pfs-offset` when every route position has
  both values (both flags true), else `z` when every one has `z_um` (Z
  only), else `none` (neither): the route's per-stop `None` (§11, #232)
  becomes one mode for the whole list, and `assumed` names how many
  positions lacked the value that decided it.
- **Round trip.** `float(f"{v:.15f}") == v` for every `|v| >= 10` (0 of
  100 000 random values failed) and not below (26 % failed in `[1, 10)`,
  off by less than 1e-15 µm). `decode(encode(x)) == x` is exact for the
  stage's values (Z and the offset are thousands, X and Y are within 10 µm
  of 0 only at the stage's centre) and within 1e-15 below 10.
- **The sample is committed as a test fixture**, under `tests/data/nis/`,
  marked `binary` in `.gitattributes` and excluded from pre-commit's
  `end-of-file-fixer` and `trailing-whitespace`: `identify` tags a `.xml`
  as text, and `end-of-file-fixer` appended one byte to the sample
  (8847 bytes, an odd length that no UTF-16 decoder reads), measured with
  the hook's v6.0.0.

Not verified: that NIS-Elements 6.20 loads a file this encoder wrote, in
each mode (a `pfs-offset` list sets both flags, which the sample does not
show), and that a stop exported this way is the field the scan imaged.

## 15. The plate page (#197)

ADR-0006, Panel, a view only. `src/smc/ui/controller.py` (no Panel
import), `src/smc/ui/plate_page.py` (Panel), and the command `smc ui
plate`.

```
smc ui plate [-p PROFILE] --run-doc PATH [--port 5006] [--dry-run]
```

It checks `--run-doc` as `smc run` does (§1.10, step 2), opens the stand
once for the page's life (one connection at a time), serves the page with
`panel.serve` on `localhost`, and closes the stand when the server stops.
Without the `ui` extra it exits 1 with `install the ui extra: pip install
-e ".[ui]"` (§15.3 fixes the order of its checks and its Ctrl-C).

### 15.1 The controller

```python
class PlatePage:
    def __init__(self, microscope: Microscope, run_doc: Path) -> None: ...

    selection: tuple[str, ...]
    refs: tuple[WellRef, ...]
    walls: tuple[WallTouch, ...]

    def view(self) -> PlateView: ...
    def select_text(self, text: str) -> None: ...
    def toggle(self, well: str) -> None: ...
    def capture_centre(self, well: str) -> WellRef: ...
    def capture_wall(self, well: str, side: str) -> WallTouch: ...
    def run(self, plugin: str, params: Sequence[str]) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> PageStatus: ...
    def close(self) -> None: ...
```

- **Every action is a plugin run with the CLI's strings**: `run(plugin,
  ["key=value", ...])` goes through `load`, `parse_params` and
  `run_plugin(..., stop=<event>, on_progress=<queue>.put)` in one worker
  thread, so each button has its `smc run` equivalent (principle 5). A
  second `run` while one is running raises `RuntimeError` ("<plugin> is
  running"). Errors are kept as their one-line text in `status()`, never
  as the exception (FM-98).
- `select_text` uses `parse_wells`; `toggle` adds or removes; the
  selection becomes `scan-plan`'s `wells` (`A1,B3,...`). It starts from
  the latest `scan-plan` record's `params`.
- `capture_centre` and `capture_wall` read `XYStage.position_um()` and
  append to `refs`/`walls`, which start from the latest
  `plate-registration` record's `params`; **Calibrate** is `run(
  "plate-registration", [plate=..., refs=<JSON>, walls=<JSON>])`, **Go
  to** is `goto-well`. Since #295 a capture is refused while an activity other
  than the page's live goes, or while a drive action waits or runs during live,
  so the position it records is where the stage rests (m5 §6.3).
- `stop()` sets the stop event, then `microscope.stop()` (the emergency
  stop of M1 §13), in that order.
- `view()` is a pure snapshot built from a fresh `RunDocument.load_or_new`:
  every well (`well_geometry` of the `plate` slot, or of the nominal plate
  before one exists), the selection, the density per label when `density`
  is usable, the route's points when `route` is, and the checklist of the
  slots of §2.2 with `unusable`'s reason for each. The stand's state (the
  stage position, the objective, the autofocus state) is `status()`'s, not
  `view()`'s: they refresh at different rhythms (§15.3).

### 15.2 The page

One page, ADR-0006's layout: the status strip (XY, objective, PFS, light,
**STOP** always visible); the plate map, a Bokeh figure in stage µm with
Y up (wells as circles or squares, selection filled, density as colour by
a chosen label, the route as a line, the stage as a cross, names); click
modes *select* (default) and *drive* (`goto-well`, an explicit toggle,
`nikon-control`'s "armed mode"); the tools in workflow order, each with
its key parameters and an *advanced* fold from `describe_params`, a Run
button and the progress; the run document as a checklist. A periodic
callback every second (§15.3) reads `status()`
on the server's thread (Bokeh models are not touched from the worker). A
browser session that ends (a closed or reloaded tab) does nothing: the
controller belongs to the server, not to a tab (§15.3). When the server
stops, `close()` runs `stop()` if a run is going, then nothing else; the
stand closes with the command. No turret control, no Z control: the page
offers none in M3.

**Tests**: the controller on `fake_microscope` and the demo, with stub
plugins through `EntryPoint`s (§1.12): `test_a_click_and_a_typed_range_build_the_same_selection`;
`test_run_goes_through_the_same_parameters_as_the_cli`;
`test_a_second_run_while_one_runs_is_refused`; `test_stop_sets_the_event_then_stops_the_stand`;
`test_captured_references_reach_plate_registration`;
`test_the_view_shows_why_a_stage_is_unusable`;
`test_a_drive_click_on_another_plates_map_moves_nothing`. The page itself:
`test_the_page_builds_on_the_demo` with `pytest.importorskip("panel")`;
CI's `test` job installs `.[dev,ui]` (#197 edits the workflow; `lint` keeps
`.[dev]`, so that `mypy` is proved without the extra, §15.3). **Failure modes**:
FM-98, FM-70 (the stop path takes no lock the worker holds), FM-44.

### 15.3 Refined while planning #197

Measured 2026-10-28 on throwaway prototypes that are not committed: Panel
1.10.0 with Bokeh 3.10.1 on Python 3.13.0 and Panel 1.9.4 with Bokeh 3.9.2 on
Python 3.10.10 (what `panel>=1.4` resolves to on each today), macOS,
`useq-schema` 0.9.2.

- **Without a browser** these work: `figure.on_event(Tap, f)` is called by
  `figure._trigger_event(Tap(figure, x=12.5, y=-3.0, sx=1, sy=2))` with the
  data coordinates; `source.data = {...}` replaces a source's data with no
  document; `pn.Column(...).get_root()` builds the Bokeh model; `button.clicks
  += 1` calls an `on_click` handler; `pn.state.add_periodic_callback(f,
  period=250, start=False)` returns a `PeriodicCallback`.
- **Serving**: `pn.serve({"": factory}, port=p, address="127.0.0.1",
  show=False, threaded=True)` returns a `StoppableThread`; `GET /` answers
  200 (11 to 16 kB) and a page of a button and a text holds no external URL;
  `stop()` returns at once, and on 3.10 `is_alive()` is still true right after
  it, so `join(timeout=10)` follows. **A taken port is silent**: the thread
  dies with an `OSError` traceback on stderr and `serve` has already returned.
- **A blocking `pn.serve` swallows Ctrl-C.** Bokeh installs its own SIGINT
  handler: `serve` returns normally (exit 0), no `KeyboardInterrupt` reaches
  the caller, and nothing in the caller says that the operator interrupted.
  With `threaded=True` and the main thread in a loop of `thread.join(0.1)`,
  SIGINT is a `KeyboardInterrupt` in the main thread (exit 130, the server
  stops in 0 s). Both Pythons, POSIX; Windows not measured (the 0.1 s slice is
  M1's FM-83 remedy).
- **Cost of a plate.** `well_geometry` for every well takes 0.2 ms (6 wells),
  1.7 ms (96), 22 ms (384) and **339 ms (1536)**; `all_well_positions` 5 ms for
  1536. 96-well wells are round; 384- and 1536-well wells are squares
  (`circular=False`). A view rebuilt every 250 ms would spend the 1536-well
  time four times a second.
- **mypy.** `mypy --strict` passes with Panel installed and not when `panel.*`,
  `bokeh.*` and `param.*` are `ignore_missing_imports` with `follow_imports =
  "skip"` (as torch, §9.0); without it, `pn.widgets.Button(...)` is a
  `no-untyped-call` with Panel installed and `import-not-found` without.
  `sys.modules["panel"] = None` makes `import panel` raise `ModuleNotFoundError`.
- **Names that warn.** Panel 1.9.4 and 1.10.0 raise a `PendingDeprecationWarning`
  for a widget's `name=` and for `Button(button_type=...)`. With `label=` and
  `color="danger"`, and with the figure, `circle`, `rect`, `scatter`, `line`,
  `LabelSet`, `ColorBar`, `LinearColorMapper(nan_color=...)`, `Accordion`, `Card`,
  `Toggle`, `Select`, `TextInput`, `pane.Markdown` and `pane.Bokeh`, a prototype
  builds under `warnings.simplefilter("error")` on both. The page uses those
  spellings, and its tests run with `PendingDeprecationWarning` as an error.
- **Not measured**: a real browser (the click that raises `Tap`, the periodic
  callback of a session, the look), Windows, RDP, a PC with no internet, and
  the cost of `Microscope.state()` on a real stand. The one-second tick runs
  it on Bokeh's server loop, the loop that also handles a click on STOP: a
  device read that hangs in a vendor SDK would delay the STOP click, whatever
  the lock (`state()` takes none, FM-16). The first hardware session times the
  tick on the Ti2-E.

**The types** (`smc/ui/controller.py`, no Panel import):

```python
@dataclass(frozen=True, slots=True)
class WellView:
    well: str
    center_um: tuple[float, float]
    size_um: tuple[float, float]
    circular: bool
    rotation_deg: float
    selected: bool
    counts: dict[str, int] | None
    per_mm2: dict[str, float] | None
```

`counts` and `per_mm2` are the well's `DensityWell` values per label, `None`
when `density` is not usable or does not list the well; `per_mm2` is also
`None` where the density is unknown (area 0).

```python
@dataclass(frozen=True, slots=True)
class RouteStop:
    order: int
    well: str
    x_um: float
    y_um: float
```

```python
@dataclass(frozen=True, slots=True)
class CheckItem:
    slot: str
    optional: bool
    state: Literal["ok", "not run", "unusable"]
    reason: str | None
```

`state` is `not run` when the slot has no provider and `unusable` when
`RunDocument.unusable` gives a reason (its text, as it is); `reason` is `None`
for `ok`. The slots, in this order: `camera-calibration` (optional), `plate`,
`timings` (optional), `scan-plan`, `scan`, `cells`, `density`, `route`,
`report`, `nis-export`.

```python
@dataclass(frozen=True, slots=True)
class PlateView:
    plate: str
    nominal: bool
    wells: tuple[WellView, ...]
    labels: tuple[str, ...]
    route: tuple[RouteStop, ...]
    checklist: tuple[CheckItem, ...]
    blocked: dict[str, str]
    problems: tuple[str, ...]
```

`nominal` is true when the wells are those of the plate type with A1 at (0, 0)
and rotation 0, not a registered plate. `labels` are `DensityResult.labels`
(`()` without a usable density). `blocked` maps a registered tool's name to why
it cannot run now: the first `unusable` reason of a slot in its `reads`, else
`this stand has no <Capability>` for the first of its `requires` that
`microscope.has` refuses, else the **checkpoint** reason below; a tool that can
run is absent. `problems` are one-line
texts of what `view()` could not read and did not hide: `run document: <the
RunDocumentError>` (the view then shows `RunDocument.new()`: every slot `not
run`, the nominal plate), and `<slot>: result does not validate: <first
error>` for a usable slot whose result its model refuses (FM-125); `view()`
raises nothing for a document or a result.

```python
@dataclass(frozen=True, slots=True)
class ToolView:
    name: str
    summary: str
    stage: str
    key: tuple[tuple[str, str, str, str], ...]
    advanced: tuple[tuple[str, str, str, str], ...]
```

The rows are `describe_params` rows (name, type, default, notes) without the
keys the page fills itself; `key` holds the keys of `KEY_PARAMS[name]` in that
order, `advanced` the rest in declaration order. A tool is shown only when the
registry holds it and `STAGES` names it (a plugin the table does not name is
not on the page in M3):

```python
STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Set up", ("camera-calibration", "plate-registration", "plate-finding", "timing")),
    ("Choose", ("scan-plan",)),
    ("Acquire", ("scan",)),
    ("Find", ("cells", "density", "route")),
    ("Report", ("report", "nis-export")),
)
KEY_PARAMS: dict[str, tuple[str, ...]] = {
    "camera-calibration": ("step_um",),
    "scan-plan": ("objective", "overlap", "coverage"),
    "scan": ("autofocus",),
    "cells": ("detector", "score_threshold"),  # detector: #260
    "route": ("labels", "fields_per_well"),
}
PAGE_PARAMS: dict[str, tuple[str, ...]] = {
    "plate-registration": ("plate", "refs", "walls"),
    "scan-plan": ("wells",),
}
```

```python
@dataclass(frozen=True, slots=True)
class PageStatus:
    dry_run: bool
    running: str | None
    progress: Progress | None
    outcome: str | None
    stop_note: str | None
    xy_um: tuple[float, float] | None
    z_um: float | None
    objective: str | None
    autofocus: str
    light_on: bool | None
    moving: tuple[str, ...]
    halted: bool
    stop_refused: tuple[str, ...]
    problems: tuple[str, ...]
```

`xy_um` to `moving` and `halted` are `Microscope.state()`'s (a read that takes
no lock and tolerates a failed device: its `errors` become `problems`);
`light_on` is `LightSource.is_on()` (`None` unknown or no light source);
`autofocus` is one of `no autofocus`, `autofocus ?`, `autofocus off`,
`autofocus on, not locked`, `autofocus locked` (the words of
`Microscope.describe`); `stop_refused` are the roles of the profile's
`[safety] stop_refused_by`, so the page can say that STOP does not stop a move
in flight on this stand (FM-117).

**What §15.1 gains** (additive; every name below is new):

```python
class PlatePage:
    def __init__(
        self,
        microscope: Microscope,
        run_doc: Path,
        *,
        entry_points: Iterable[EntryPoint] | None = None,
    ) -> None: ...

    current: str | None

    @property
    def plate(self) -> str: ...
    def choose_plate(self, name: str) -> None: ...
    def tools(self) -> tuple[ToolView, ...]: ...
    def click(
        self, x_um: float, y_um: float, *, drive: bool, side: str = "centre"
    ) -> str | None: ...
    def goto(self, well: str, side: str = "centre") -> None: ...
    def forget(self, well: str | None = None) -> None: ...
    def page_params(self, plugin: str) -> list[str]: ...
    def resume(self) -> None: ...
    def join(self, timeout_s: float | None = None) -> bool: ...
```

- **`entry_points`** is handed to `discover` and `load`; `None` reads the
  installed group. It is how a test puts stub plugins on the page (§1.12).
- **`plate`** starts as the latest `plate-registration` record's
  `params["plate"]`, else the usable `plate` slot's `plate`, else `96-well`;
  `choose_plate` takes a name of `useq.registered_well_plate_keys()` (else
  `ValueError` with the names) and, when it differs, clears `selection`,
  `refs`, `walls` and `current`. The map draws the `plate` slot's wells only
  when it is usable **and** its `plate` equals `self.plate`; otherwise it draws
  the nominal wells (`nominal=True`). The wells of a registered plate are built
  once per `(plate, a1_center_xy_um, rotation_deg)` and kept (339 ms for 1536
  wells): `view()` and `click` reuse them.
- **`click(x_um, y_um, drive=, side=)`** finds the well whose `WellGeometry`
  contains the point (none: returns `None` and does nothing, as for a point
  that is not finite), sets `current`, and then `toggle`s it, or, with
  `drive`, `goto`es it. It returns the well. The mode is the page's: `drive`
  is **one click**, and the page returns to *select* after it.
- **`goto(well, side)`** is `run("goto-well", [f"well={well}", f"side={side}"])`
  and sets `current`. **It refuses a plate the map does not show.** `goto-well`
  takes a well's name only and reads the registered plate (`reads = ("plate",)`,
  `well_geometry(plate.to_plan(), params.well)`), while the map of a plate the
  operator chose with `choose_plate` is the nominal one when the usable `plate`
  slot names another: a click on C5 of the nominal 96-well map would drive the
  stage to C5 of the registered 384-well plate, with nothing on the page to
  say so. So `goto`, and `click` with `drive`, raise the `ValueError` of
  `page_params("scan-plan")` for that mismatch (the same check, one private
  helper), before `current` changes and before a thread starts. An unusable
  `plate` slot needs no check here: `check_ready` refuses `goto-well` itself and
  the refusal is the `outcome`. **`capture_centre(well)`** and `capture_wall(well, side)`
  normalise the well (`normalise_well`), refuse one that is not on `plate`
  (`ValueError`: `<well> is not a well of the <plate> plate`), and **replace**
  the earlier capture of the same well (and side; `side` is an image side
  and the touch's a stage side, m5 §13): the operator centres it
  again, and `plate-registration` refuses a well given twice. `forget(well)`
  drops the captures of one well, `forget()` all of them.
- **`select_text`** and `toggle` take the wells of `plate`; a text or a well
  that is not on it raises `ValueError` (`parse_wells`' message; for `toggle`,
  `<well> is not a well of the <plate> plate`) and leaves `selection` as it
  was. The page shows the message and keeps no exception (FM-98).
- **`page_params(plugin)`** gives the pairs the page fills itself, named by
  `PAGE_PARAMS`: for `scan-plan` `[f"wells={','.join(selection)}"]`, and
  `ValueError("select at least one well first")` when there is none, and
  `ValueError("the registered plate is a <p>; calibrate the <plate> plate
  again, or choose <p>")` when a usable `plate` slot names another plate than
  `plate`; for `plate-registration`, `[f"plate={plate}", f"refs={json}",
  f"walls={json}"]` with the captures dumped by `model_dump(mode="json")` as
  `json.dumps` (a single line); any other plugin `[]`. The page runs
  `run(plugin, page_params(plugin) + typed)`, `typed` being the form's
  non-empty fields as `key=value`.
- **`run(plugin, params)` validates before it starts a thread**, with what
  `smc run` does before it opens the stand: `load`, `parse_params`,
  `RunDocument.load_or_new`, `check_ready(..., resume=False)`. Any
  `PluginError` or `RunDocumentError` there is an `outcome`
  (`<plugin>: <Type>: <message on one line>`), nothing is written and no thread
  starts. The only exceptions `run` raises are `RuntimeError("<plugin> is
  running")` (a run is going) and `RuntimeError("the page is closed")`. The
  worker is a daemon thread (a stuck plugin never keeps the process alive after
  `close`) running `run_plugin(..., stop=<a new Event for this run>,
  on_progress=<SimpleQueue>.put, plugin_version=entry.version)` and keeping, as
  text, `outcome`: `<plugin> done in <s:.1f> s`, `<plugin> stopped`,
  `<plugin>: <Type>: <message on one line>` for any `Exception`, and
  `<plugin> ran, but its record could not be written: <message>` for a
  `RunDocumentError` that carries a record (FM-34). There is no `resume` in the
  page: `smc run --resume` is the way to continue a stopped stage in M3, and
  so the page never starts a stage afresh over a checkpoint that `--resume`
  could still reach (#252):
  - **The checkpoint refusal.** `smc.plugins.runner` gains
    `reachable_checkpoint(name: str, doc: RunDocument, *, profile: str,
    dry_run: bool) -> StageStatus | None` (in `__all__`; it touches neither
    hardware nor files, like `check_ready`). It returns the record's status
    (`stopped` or `failed`) when the stage's record has one of those statuses,
    a `checkpoint` that is not `None`, and `--resume` would not refuse it for
    what it knows without the new parameters: the record's `profile` is
    `profile`, it is not a dry run's while this run is not (`record.dry_run and
    not dry_run`), and `doc.is_stale(name)` is false (the three tests of
    `check_ready` that need no `params`); otherwise `None`. A checkpoint that
    `--resume` refuses anyway (another profile, a dry run's, stale) is not
    protected: nothing is lost by replacing what no command can reach. The
    parameters are **not** compared: the page's parameters are the form's and
    change between clicks, and a stage that was stopped on purpose is lost by a
    changed `step` as surely as by an unchanged one. A `running` record is not
    covered (it is the run itself, or a process that died; `unusable` already
    says `still running or was interrupted`).
  - Its text, built by the controller (`_checkpoint_reason(name, status)`), is
    ``<status> with a checkpoint: continue it with `smc run <name> --resume`
    (close this page first); `smc run <name>` without --resume starts afresh and
    drops it``, `<status>` being `stopped` or `failed`. It says no path: the
    operator's run document is the one this page was started with, and
    `--run-doc` is required by the command whatever the text says.
  - **In `view()`** it is the last reason of `blocked[name]` (after `reads` and
    `requires`: a tool that cannot run at all says why first), so the Run
    button is disabled with that text beside it and nothing else in the page
    changes. **In `run()`**, after `check_ready` and before a thread starts, a
    tool with such a record is an `outcome` `<plugin>: <the text>`: nothing is
    written, and the stale tab whose button was still enabled, or a caller of
    the controller, gets the same refusal. `smc run` without `--resume` is
    **unchanged**: it starts afresh and logs that it does (§1.8); it is the
    deliberate way to give a checkpoint up, and the page does not offer it.
  - **Tests**: `test_run_refuses_a_stage_stopped_with_a_checkpoint_and_says_how_to_resume`,
    `test_a_failed_stage_with_a_checkpoint_is_refused_too`,
    `test_a_checkpoint_resume_would_refuse_does_not_block_a_run` (another
    profile, a dry run's, a stale record: one case each),
    `test_a_stopped_stage_without_a_checkpoint_runs_afresh`,
    `test_view_blocks_a_stage_with_a_reachable_checkpoint_after_its_other_reasons`,
    `test_the_run_button_of_a_stopped_stage_is_disabled_with_the_resume_command`
    (the page), and `reachable_checkpoint`'s own cases in `test_plugin_runner.py`.
- **`status()`** drains the progress queue (the latest `Progress` is kept, so
  two sessions see the same) and reads the stand now; the page's periodic
  callback calls it once a second (`period=1000`) and no faster: a scan must
  not be slowed by a status poll, and the poll's cost on a real stand is not
  measured. **`view()`** runs on an event (a click, a typed range, a capture, the
  end of a run) and on a *Refresh* button, never on the tick.
- **`stop()`** sets the run's stop event (a plain attribute read, then
  `Event.set()`; no lock of the controller, FM-70), then `microscope.stop()`,
  and keeps what it did as `stop_note`, said by its outcome and never "stopped"
  by default (FM-117): `stopped: the stand is halted (Resume clears the halt)`;
  `halted, but <the StopRefusedError, on one line>`; `the stop failed: <the
  HardwareError, on one line>`; each starts with `<plugin> told to stop;`
  and a space when a run was going. It never raises for a stop that fails. STOP is always
  offered, a run or not. **`resume()`** refuses while a run is going
  (`RuntimeError("<plugin> is running")`), then `microscope.resume()` and clears
  `stop_note`; the page offers **Resume** only while `status().halted`, because
  after STOP every action raises `MicroscopeHaltedError` until the halt is
  cleared. **`join(timeout_s)`** waits for the worker and says whether none is
  left (tests, `close`). **`close()`** is idempotent: when a run is going,
  `stop()`, then `join(10.0)`; if the worker is still alive it says so in
  `outcome`. It does not close the stand: the command owns it.
- **The lock** the controller holds (`_lock`, a plain `Lock`) guards only the
  attributes a worker and a callback both write (`running`, `progress`,
  `outcome`), never across a device call, and `stop()` does not take it.

**The command** `smc ui plate`, in `smc/cli.py` (`smc.ui.*` is imported inside
the command, so `smc` starts without the extra), in this order:

1. `--run-doc` as `smc run` checks it (exit 2); `--port` an integer in 1 to
   65535, default 5006 (exit 2).
2. `import smc.ui.plate_page`: an `ImportError` (the extra is missing, or
   broken) prints `install the ui extra: pip install -e ".[ui]"` and the error's
   text, exit 1.
3. A socket binds `localhost:<port>` and is closed again: when that fails,
   `port <port> is in use; choose another with --port` (exit 2). `serve` below
   checks again, since between the two another process may take the port.
4. `Profile.load`, then `RunDocument.load_or_new` (an unreadable document: its
   text, exit 1, the file untouched), then `Microscope.open(profile,
   dry_run=dry_run)`, then `_say_stop_refused` unless `--dry-run`.
5. `PlatePage(microscope, path)`, then `plate_page.serve(page, port=port)`,
   which builds the layout per session, serves with `pn.serve(...,
   address="localhost", show=False, threaded=True)` and waits in a loop of
   `thread.join(0.1)`. It prints `serving the plate page on
   http://localhost:<port>/ — Ctrl-C stops it` once the thread is up. On
   `KeyboardInterrupt` it stops the server and `join(timeout=10)`s it, and
   returns; if the thread ended **without** Ctrl-C it raises
   `PageServerError` (a `RuntimeError` in `smc/ui/controller.py`: `the plate
   page's server stopped on its own (is port <port> taken?)`), which `_report`
   prints with exit 1.
6. `page.close()` in a `finally`, then the stand is released by the `with`;
   exit 0 after a Ctrl-C (a server's normal end), the last line saying
   `<plugin> was stopped` when `close()` stopped a run. `serve` takes the
   first Ctrl-C; a second one, during `close()`'s join, is the `_Reported`
   wrapper's own `KeyboardInterrupt` path (its interrupted line, exit 130,
   FM-71). It loses neither the stop, which `close()` sent **before** it
   joined, nor the release of the stand, which the `with` does on any exit.

The page listens on `localhost` only and takes no `--host`: it has no login, so
whoever reaches the port can move the stage. A desk is reached by RDP in M3. An
SSH tunnel is not tried: Bokeh checks a websocket's origin against
`localhost:<port>`, so it can work only with the same port number on both ends,
and that is not verified; `docs/ui/plate-page.md` says so or leaves it out.

**The page** (`smc/ui/plate_page.py`, the only module besides the command that
imports Panel). Decided so that the executor guesses nothing:

- `build(page: PlatePage) -> pn.viewable.Viewable` builds one session: the same
  controller, new widgets. A `PlatePageView` class holds the widgets and the
  figure, with `tick()` (the periodic callback's body, called by tests
  directly), `refresh()` (`view()` then redraw) and one method per handler.
  `build` registers `pn.state.add_periodic_callback(view.tick, period=1000)`.
- The map: `bokeh.plotting.figure(match_aspect=True)`, stage µm, Y up. One
  `ColumnDataSource` for the wells with the columns `name, x, y, w, h, angle,
  value, outline, outline_width, tip` (`angle` in radians, counter-clockwise,
  `rotation_deg` converted; `w` and `h` are `size_um`; a circle takes
  `radius = w / 2`), drawn as `circle` when `circular` and `rect` otherwise;
  `value` is the chosen label's `per_mm2` and `nan` where unknown, mapped by a
  `LinearColorMapper(palette=Viridis256, low=0, high=max, nan_color="#eeeeee")`
  with a `ColorBar`; a selected well has `outline_width=3` in blue and the
  others `1` in grey, so a selected well keeps its density colour. The names
  are drawn when there are at most 96 wells. The route is a `line` and a
  `scatter` of its stops, the stage a `cross` at `status.xy_um`.
- The click: `figure.on_event(Tap, handler)` calls `controller.click(event.x,
  event.y, drive=<toggle>, side=<select>)`, shows an exception's message and
  puts the toggle back to *select*; while *drive* is on the status strip says
  so in the warning colour. The tools: one `Accordion` card per `STAGES`
  entry, one section per tool with its `key` fields, an *Advanced* card with
  the rest, a Run button (disabled, with `blocked[name]` beside it, when the
  tool is blocked or a run is going) and the progress as `done/total message`;
  an empty field is not sent, so the plugin's default holds. `plate-registration`
  also has the plate choice, **Capture centre**, **Capture wall** (a side
  choice) and **Forget** on `current`, and the captures as a list.
- The status strip: XY, Z, objective, autofocus, light, `DRY RUN` when it is
  one, `HALTED` with **Resume** when it is, `outcome`, `stop_note`, the run's
  progress and **STOP**, in the order of ADR-0006's picture. The checklist is
  `view.checklist` as `✓ <slot>`, `· <slot>` and `✗ <slot>: <reason>`. A
  `problems` text is shown above the map, in the warning colour.
- **Tests of the page** call `PlatePageView` methods and read the models:
  `wells_source.data`, `figure.renderers`, the widgets' `value` and `disabled`,
  `source.selected` is not used.

**Dependencies**: `ui = ["panel>=1.9,<2"]` (the oldest measured is 1.9.4 on
Python 3.10; a floor below what CI resolves would be a claim nothing proves);
CI's `test` job installs `.[dev,ui]` and sets `SMC_REQUIRE_UI`; `lint` keeps
`.[dev]`, so that `mypy` is proved without the extra, as a contributor who does
not install it runs it; `pyproject.toml` gains the mypy override above. The
page's HTML is not checked offline.

**Failure modes**: FM-98, FM-70, FM-44, FM-117, FM-125, FM-30, FM-36, FM-37,
FM-20, FM-24, FM-71, FM-73, FM-45, FM-50, and the entries the develop session
adds for what this section measured: a server that swallows the operator's
Ctrl-C, a port taken that no one is told of, a refresh that rebuilds an O(wells)
structure on every tick, and a view lifetime that is not the work's lifetime
(a closed tab must not stop a scan).

## 16. Sequencing

1. **#199, the foundation (§3)**, first and alone: every other issue
   imports it.
2. Then, in parallel, each in its own package (no two issues touch the
   same module): **#22** (§4), **#19** (§5), **#21** (§7), **#213**
   (§8), **#23**
   (§9), **#24** (§12), **#193** (§10), **#194** (§11), **#20** (§6).
   Shared files: each adds its entry point line to `pyproject.toml`
   (sorted by name; the rebase conflict is expected and trivial), #22 adds
   `scikit-image`, #23 the `detect` extra, #197 the `ui` extra.
3. **#195** (§13) and **#197** (§15) once the plugins they show exist; both
   build on stubs and can start earlier.
4. **#196** (§14) once the NIS sample is in `local/`.
5. **#198**, session 2 at the stand.

For session 2 the critical path is #199 → #22, #19, #21, #213, #23 →
#194 →
#195 or #197 → #196.

### 16.1 Not verified (for #198 and the owner)

- The Ti2-E's pixel size at 40× (0.275 µm assumed) and the camera's
  rotation; the stage's repeatability (§4's return check measures it).
- Whether the PFS holds its lock across a 96-well well at 40× in
  brightfield, and how often `engage()` is needed per field.
- The model's behaviour on Prime 95B frames: the frames it was trained on
  are not recorded (size, pixel size and camera, §9.4; the "1024²" first
  written here came from an illustrative shape), and it resizes everything
  to 800 px, so a 1412² frame is seen at about 0.49 µm per pixel. Measured
  on the first real scan with §9.4's script (#342).
- Inference time on the microscope PC's CPU (0.34 s per frame on an M5
  Pro); a well of 305 frames would take about 2 minutes at that speed.
- `merge_radius_um = 5.0` and the overlap of 10 % against real cells.
- Whether the stage's move time follows the Euclidean distance (§11).
- That NIS-Elements 6.20 loads the export in each of its modes, and that
  its stage coordinates are Micro-Manager's (§14).
- Panel's version and behaviour on the microscope PC over RDP; a real
  browser's click (the `Tap` event of §15.3), the periodic callback of a
  session, Ctrl-C on Windows, and the page on a PC with no internet.

## 17. Plate registration from images of the walls (#328)

`smc/plugins/plate_walls/algorithm.py` and `plugin.py`, entry point
`plate-walls = "smc.plugins.plate_walls.plugin:PlateWallsPlugin"`. It fills
`plate` with `method="wall-images"`, so nothing downstream changes.

**What the operator does** (the owner's form, 2026-11-05): with the joystick,
put the stage roughly at the centre of a well, and record that position for
that well; do it for 2 or 3 wells far apart. The tool does the rest: around
each recorded position it images the well's four walls, fits a circle to the
edges (the well's centre and diameter), and fits the plate to the centres.
Recording a position is a stage read (`smc stage get`, or the page's capture,
§17.5); the tool moves XY only, around the recorded positions.

**Not here**: a stage put in the first well only, the tool driving to the
others from the nominal pitch (the owner's second step, once this one works:
it drives blind on a plate whose rotation is unknown); square wells (`384-well`,
`1536-well`); a mode of `plate-registration` (that plugin moves nothing and
runs in a dry run; this one moves and cannot); measuring the camera on the
spot. `plate-finding` stays for the hands-off survey.

### 17.1 What was measured (2026-11-05, planning #328)

On throwaway prototypes that are not committed; numpy 2.5.3, tifffile
2026.9.20, useq-schema 0.9.2, Python 3.13.

- **The B3 frames of Ti2-E session 2** (`local/surveys/nikon-ti2/session2/`,
  private: Prime 95B, 1412 × 1412 px, 10×, brightfield, a 24-well plate,
  #334). The file names are the side on the *image*; the stage positions are
  `walls.txt`'s.

  | file | stage XY (µm) | wall on the stage's side | edge from the centre (px) |
  |---|---|---|---|
  | `B3-right.tif` | (2468.0, −9737.3) | −X | −29 (column) |
  | `B3-left.tif` | (18067.4, −9737.3) | +X | +183 (column) |
  | `B3-bottom.tif` | (10226.7, −2337.6) | +Y | −122 (row) |
  | `B3-top.tif` | (10226.7, −17738.5) | −Y | +28 (row) |

  - **The camera at 10× is turned 180° and not mirrored.** The wall the
    operator saw on the image's right is the −X one, and the image's bottom
    is +Y: a +column offset is −X and a +row offset is +Y, `M = 1.1 ·
    ((−1, 0), (0, 1))` µm per pixel (`det M < 0`: not mirrored, §3.1;
    rotation 180°). This is read from the frames and the stage positions;
    `camera-calibration` was not run at 10× (step 7 of the session was not
    reached), and 1.1 µm per pixel is the nominal size.
  - **The edges.** Each frame's dark class (below the midpoint of its 2nd
    and 98th percentiles) covers 53–64 % of it, at 0.057–0.093 of the bright
    class's mean. The 2nd percentile is 157–198 counts and the 98th
    9276–10 372. Inside the well the level falls towards the wall (9000 →
    6400 → 1750 → 200 counts over 700 px of `B3-right.tif`): the meniscus.
    The centre frame `B3-center.tif` reads 31 000–35 500.
  - **The fit.** §17.3's `find_wall` gives 173–176 edge points per frame (of
    176 lines) in 15 ms each. `fit_circle` on the four frames' 698 points
    keeps 696: centre (10 182.0, −10 087.9) µm, diameter 15 298 µm (nominal
    15 600, −1.9 %), rms 41.9 µm. The rms is the well's shape, not noise: the
    X walls lie 40 µm outside the circle and the Y walls 38 µm inside it
    (wall to wall, 15 366 µm in X and 15 236 µm in Y).
  - **Against the walls' midpoint** of the session's own table (#334: the
    half-intensity edges −30, +181, −123 and +26 px, with the matrix above),
    (10 184.7, −10 091.4) µm: the fitted centre is 4.4 µm away.
  - **The camera's turn matters.** With the matrix of an upright camera
    (rotation 0, `pixel_matrix(1.1)`), `find_wall` finds no wall in any of
    the four frames: the outward direction points into the plastic, every
    line starts dark, and no line gives a point. A wrong calibration is a
    refusal, not a centre one well-width off (#327's failure).
- **Synthetic wells** (`PlateSample.render`, no stage), the whole walk of
  §17.4 with §17.3's functions, rough centres off by up to half a radius:

  | plate | px (µm) | frame | camera | off by (µm) | frames | worst centre error |
  |---|---|---|---|---|---|---|
  | 96-well, 1.2° | 0.65 | 384 × 512 | 30°, mirrored | (300, −200) | 135 for 3 wells | 0.01 µm |
  | 96-well, −3° | 1.625 | 512 × 384 | −75° | (−900, 700) | 48 for 2 | 0.07 µm |
  | 24-well, 2° | 1.1 | 1412 × 1412 | 180° | (1500, −1200) | 72 for 3 | 0.00 µm |
  | 24-well | 2.75 | 1412 × 1412 | 90°, mirrored | (−2500, 2000) | 26 for 2 | 0.01 µm |
  | 6-well, 0.5° | 1.625 | 512 × 512 | 10° | (3000, 0) | 76 for 2 | 0.09 µm |
  | 48-well, 0.5° | 1.625 | 512 × 512 | 10° | (1000, −1000) | 60 for 2 | 0.04 µm |

  The diameter came back within 0.07 µm and the rms within 0.3 µm. Those
  frames walked from the inside outwards. §17.4's alternating order took 14
  frames per well for the first case (45 before), and a calibration 2 % too
  large and 0.5° off, or 2 % too small and −0.5° off, moved the centres by
  0.06 µm at most: the centred frame puts the wall where the matrix matters
  least. `find_wall` also finds the wall under noise of 600 counts (well
  3000, plastic 800), and finds none in: a frame wholly inside a well (the
  texture's dark ratio 0.72), a linear ramp from 6000 to 9000 counts (0.82,
  the meniscus without the wall), a frame on the plastic between wells (1
  line of 48), and a frame whose only edge is the neighbouring well's near
  wall (0 of 48: its lines start dark). On a 48-well plate at 5 µm per
  pixel, a frame holding both our wall and the neighbour's took ours (971 µm
  from the frame's centre, the wall at 950).
- **A plate of another pitch.** A synthetic 96-well plate of pitch 9.03 mm,
  fitted as the 96-well plate (9.0): residuals of 196 µm (A1, H12), 197 µm
  (A1, H12, D7) and 231 µm (A1, A12, H1), scale error +0.0033, below
  `SCALE_TOLERANCE`. The residual bound of §17.4 refuses it.
- **useq's round plates** (0.9.2): the 6-, 12-, 24-, 48- and 96-well plates,
  with gaps between wells (pitch minus diameter) of 4.32, 4.0, 3.4, 1.9 and
  2.6 mm. The 384- and 1536-well wells are square.

### 17.2 The slot models — `slots/plate.py`

```python
WallSide = Literal["+X", "+Y", "-X", "-Y"]


class WallFrame(SlotValue):
    side: WallSide
    x_um: float
    y_um: float
    n_points: int = Field(ge=0)
    frames: int = Field(ge=1)
    file: str | None = None
```

```python
class WellFit(SlotValue):
    well: str
    recorded_xy_um: tuple[float, float]
    center_xy_um: tuple[float, float]
    diameter_um: float = Field(gt=0.0)
    size_error: float
    rms_um: float = Field(ge=0.0)
    n_points: int = Field(ge=0)
    walls: tuple[WallFrame, ...]
```

```python
class WallImages(SlotValue):
    objective: str | None
    pixel_size_um: float = Field(gt=0.0)
    image_shape_px: tuple[int, int]
    search_um: float = Field(gt=0.0)
    wells: tuple[WellFit, ...]
    frames: int = Field(ge=0)
    folder: str | None
    elapsed_s: float = Field(ge=0.0)
```

- **Sides are stage directions**, never image sides: `+X` is the wall the
  stage reaches by going +X from the well's centre. No operator names a side
  on this path, so #327's confusion cannot arise here.
- `WallFrame` is the frame whose edge points were used for that side (the
  one centred on the wall, §17.4): `x_um`, `y_um` are the stage's readback
  where it was taken, `n_points` its edge points, `frames` how many frames
  that side took in all (the walk and the centred one), `file` its name in
  `WallImages.folder` (`B3_+X.tif`), `None` when frames are not saved or this
  one could not be.
- `WellFit`: `recorded_xy_um` is where the operator put the stage;
  `center_xy_um`, `diameter_um`, `rms_um` and `n_points` (the points kept)
  are `fit_circle`'s; `size_error` is the diameter over the nominal, minus 1.
  `walls` in the order `+X`, `+Y`, `-X`, `-Y`.
- `WallImages`: the objective in place (`None` without a turret), the pixel
  size and frame of the calibration used, the search distance used, the wells
  in the order given, `frames` in all, `folder` (run-relative, `plate-walls/
  <token>`; `None` when frames are not saved) and the run's duration
  (`time.monotonic`, from the first move to the result).
- **`PlateResult`** gains `walls: WallImages | None = None` and the method
  `wall-images` (§3.2). For this method, `refs` are the fitted centres
  (`WellRef(well, *center_xy_um)`, the wells in the order given) and `n_refs`
  their count; `fit` is `None`.

### 17.3 Algorithm — `plate_walls/algorithm.py`

Pure: numpy only, no device, no file. No matrix product anywhere (`@`,
`.dot`, `.matmul`: FM-131, `tests/unit/test_no_matrix_product.py`); the 2 × 2
products and the inverse are written out.

```python
BAND_PX = 8
MIN_CLASS_FRACTION = 0.02
MAX_DARK_RATIO = 0.5
MIN_LINE_FRACTION = 0.5
STEP_FRACTION = 0.75
DEFAULT_SEARCH_FRACTION = 0.5
MAX_RMS_FRACTION = 0.02
MIN_FRAME_SIDE_PX = 128
MAX_FRAMES = 600
SIDES: tuple[WallSide, ...] = ("+X", "+Y", "-X", "-Y")
OUTWARD: dict[WallSide, tuple[float, float]] = {
    "+X": (1.0, 0.0),
    "+Y": (0.0, 1.0),
    "-X": (-1.0, 0.0),
    "-Y": (0.0, -1.0),
}


@dataclass(frozen=True, slots=True)
class WallEdge:
    points_px: npt.NDArray[np.float64]
    n_lines: int
    threshold: float
    dark_ratio: float


@dataclass(frozen=True, slots=True)
class CircleFit:
    center_xy_um: tuple[float, float]
    diameter_um: float
    rms_um: float
    n_points: int
    n_used: int


def find_wall(
    frame: npt.NDArray[Any], *, outward_xy: tuple[float, float], matrix: Matrix2
) -> WallEdge | None: ...
def wall_points_um(
    points_px: npt.NDArray[np.float64],
    x_um: float,
    y_um: float,
    *,
    matrix: Matrix2,
    image_shape_px: tuple[int, int],
) -> npt.NDArray[np.float64]: ...
def along_um(
    points_um: npt.NDArray[np.float64],
    centre_xy_um: tuple[float, float],
    outward_xy: tuple[float, float],
) -> float: ...
def walk_offsets_um(
    radius_um: float, search_um: float, step_um: float
) -> tuple[float, ...]: ...
def fit_circle(points_um: npt.NDArray[np.float64], *, floor_um: float) -> CircleFit: ...
def encode_wall_frame(frame: npt.NDArray[Any], meta: dict[str, Any]) -> bytes: ...
```

- **`find_wall`** finds a well's wall in one frame, the inside bright and
  the outside dark (brightfield, §17.1), looking outwards along `outward_xy`
  (a stage direction). In order:
  1. A frame that is not 2-D is a `ValueError`. It is cast to `float64`.
     `threshold` is the mean of its 2nd and 98th percentiles; the dark class
     is the pixels below it.
  2. **Two classes or none.** `None` when the dark class covers less than
     `MIN_CLASS_FRACTION` of the frame or more than `1 −
     MIN_CLASS_FRACTION`, or when the bright class's mean is not positive, or
     when `dark_ratio`, the dark class's mean over the bright class's, exceeds
     `MAX_DARK_RATIO` (a frame inside a well, shading included, §17.1).
  3. **The lines.** `v = M⁻¹ · outward_xy` is the outward direction in pixels
     (`(d_col, d_row)`, §3.1's convention). When `|v_col| >= |v_row|` the lines
     are bands of `BAND_PX` rows (from row 0, the rows left over dropped),
     each averaged across its rows and walked along the columns in the
     direction of `v_col`'s sign; otherwise bands of `BAND_PX` columns walked
     along the rows. `n_lines` counts the bands.
  4. **A line's point.** A line counts only when the mean of its first
     `BAND_PX` samples, at the inward end, is at least `threshold`: it starts
     inside a well. Its edge is the first sample below `threshold` after the
     first, interpolated linearly with the sample before it; the point is
     `(col, row)` with the band's middle (`start + (BAND_PX − 1) / 2`) on the
     other axis. A line that never falls below gives no point. This is what
     keeps a neighbouring well's walls out: going outwards, our wall is the
     first fall from bright to dark; the neighbour's near wall is a rise, and
     a line that starts in the plastic is not read.
  5. `None` when fewer than `MIN_LINE_FRACTION × n_lines` lines give a point;
     otherwise the `WallEdge`, `points_px` of shape `(n, 2)`.
- **`wall_points_um`** maps the points to the stage, the frame taken at
  `(x_um, y_um)`: §3.1's `stage_of_pixel` over arrays (the centre `((W −
  1) / 2, (H − 1) / 2)`, the products written out), agreeing with it.
- **`along_um`** is the median, over the points, of their offset from
  `centre_xy_um` projected on `outward_xy` (a unit vector): how far out the
  wall is from the recorded centre, on that side.
- **`walk_offsets_um`** is the walk of one side, the distances from the
  recorded centre at which frames are taken until a wall is found: `r`, then
  `r + s`, `r − s`, `r + 2s`, `r − 2s`, ... while `k · s <= search_um`
  (`r` the radius, `s` the step). It raises `ValueError` for an input that is
  not finite and positive. Its length is `1 + 2 · floor(search_um / step_um)`.
- **`fit_circle`** is the algebraic (Kåsa) fit, `x² + y² = 2ax + 2by + c` by
  `np.linalg.lstsq`, then twice: the residuals `|p − centre| − radius` of
  all points, the points kept those within `max(5 × 1.4826 × MAD, floor_um)`
  (`MAD` the median absolute deviation of the kept points' residuals), and
  the fit again on them. `rms_um` is over the kept points, `n_points` the
  points given, `n_used` the points kept. It raises `ValueError` for fewer
  than 3 points, a points array not of shape `(n, 2)` or not finite, a
  rank-deficient system (collinear points), fewer than 3 points kept, or a
  radius that is not finite and positive.
- **`encode_wall_frame`** is `scan.algorithm.encode_frame`'s three lines with
  a dict as the metadata: a TIFF in memory, written whole by
  `write_atomically` (FM-38).

### 17.4 Plugin — `plate_walls/plugin.py`

```python
class PlateWallsParams(PluginParams):
    plate: str = Field("96-well", description=...)
    refs: tuple[WellRef, ...] = Field((), description=...)
    search_um: float | None = Field(None, gt=0.0, le=20_000.0)
    max_residual_um: float = Field(100.0, gt=0.0, le=1000.0)
    max_rotation_deg: float = Field(10.0, gt=0.0, le=45.0)
    settle_ms: float = Field(0.0, ge=0.0, le=10_000.0)
    save_frames: bool = True
```

`requires = (XYStage, Camera)`, `reads = ("camera-calibration",)`,
`reads_optional = ()`, `provides = "plate"`. The objective is
`microscope.get(ObjectiveTurret)`'s `current()`, `None` without a turret. No
checkpoint: a run takes about a minute, and a stopped one is run again.

- `plate` is checked by `useq.WellPlate.from_str` in a field validator (an
  unknown name is a parameter error, before the stand opens). `refs` are the
  same `WellRef`s as `plate-registration`'s (the page's captures feed both),
  given as JSON: `--param 'refs=[{"well": "B3", "x_um": 10200, "y_um":
  -9700}, ...]'`. `search_um` is how far, on each axis, the wall may lie
  from where the recorded centre puts it: the default is half the well's
  radius (3900 µm on a 24-well plate, 1600 µm on a 96-well one).
  `max_residual_um` is set here, 100 µm, and revisited at the stand (§17.7).

**Before anything moves**, in this order, each failure a `PluginRunError`
and nothing moved:

1. A dry run: `a dry run cannot image the walls; run it without --dry-run`.
2. The plate's wells are square: `plate-walls fits round wells, and the
   <plate> plate's are square: register it with plate-registration`.
3. The references: fewer than 2 (`give at least two wells in refs, each with
   the stage position where it was centred by eye`), more than 6, a well not
   on the plate (`<well> is not a well of the <plate> plate`), a well given
   twice after `normalise_well`, or a coordinate beyond ±1 000 000 µm (FM-32).
4. The calibration (`ctx.upstream("camera-calibration", CameraCalibration)`):
   its `objective` must equal the objective in place and its
   `image_shape_px` `camera.image_shape()`, with scan-plan's words for each
   (§7.3, `it is of <a>, the camera is at <b>`), and `|det M| >= 1e-12`
   (FM-125), each refused with `run camera-calibration with <objective> in
   place, on a field with texture: a well's wall will do`.
5. The frame: its smaller side under `MIN_FRAME_SIDE_PX` (`a frame of <h> ×
   <w> px is too small to find a wall in`), or the larger extent of `fov_of`
   at least the well's diameter (`the field of <f> µm holds a whole well of
   <d> µm: use a higher magnification`).
6. The walk. `r` is the nominal radius (`well_size[0] × 500`), `search_um`
   the parameter or `DEFAULT_SEARCH_FRACTION × r`; a given `search_um`
   above `r` is refused (`search_um <s> is more than the well's radius <r>
   µm`). The step is `STEP_FRACTION × min(H, W) × pixel_size_um`: the
   inscribed circle of the frame, so that it holds whatever the camera's
   turn and consecutive frames overlap by a quarter. The worst case,
   `len(refs) × 4 × (len(walk_offsets_um(r, search_um, step)) + 1)`, above
   `MAX_FRAMES` is refused (`up to <n> frames: use a lower magnification or a
   smaller search_um`; FM-74).
7. **The soft limits.** When `xy.limits_um()` is not `None`, every frame of
   every walk (the recorded centre plus `OUTWARD[side] × offset`) must lie
   within them (`low <= v <= high`, the layer's own test), else `<well>'s
   <side> wall would be looked for at (<x>, <y>) µm, outside the stage's soft
   limits`. With `None`, `assumed` gains `the stage reports no soft limits, so
   the frames' moves were not checked against any travel` (as §6.2's).

Then `xy.wait()` (FM-85), the start position is read (for the warning
below), and the clock starts.

**Each well, in the order given; each side, in `SIDES` order**:

1. **The walk.** For each `d` of `walk_offsets_um(r, search_um, step)`:
   `ctx.raise_if_stopped()`; `at = xy.move_to_um(recorded + OUTWARD[side] ×
   d)`; a readback that is not finite is refused (`the stage's readback is
   not a finite position`); `ctx.wait_s(settle_ms / 1000)`; `frame =
   camera.snap()`, whose shape must equal the calibration's (`the frame has
   shape <s>, the calibration <c>: check the camera's binning and region`);
   `find_wall(frame, outward_xy=OUTWARD[side], matrix=M)`. With an edge,
   `e = along_um(wall_points_um(..., *at, ...), recorded, OUTWARD[side])`;
   the walk ends when `|e − r| <= search_um`, and goes on otherwise. The
   frame is placed where the stage says it was, never at the target (FM-142).
2. **No wall** after the last offset: `no wall found on <well>'s <side> side
   in <k> frames, from <r − search> to <r + search> µm out of the recorded
   centre (<x>, <y>) µm: check that the light is on (the inside of a well is
   brighter than the plastic in brightfield) and that the stage was in
   <well> when it was recorded, then record it again`.
3. **The centred frame.** One more move, to `recorded + OUTWARD[side] × e`,
   puts the wall at the frame's centre; settle, snap, shape check,
   `find_wall`. No edge there, or one whose `|e − r| > search_um`: `the wall
   of <well>'s <side> side was lost in the frame centred on it at (<x>, <y>)
   µm`. Only this frame's points are kept: every side is then framed the
   same way, so that the threshold's bias (the meniscus, §17.1) is the same
   on opposite walls and cancels in the centre.
4. With `save_frames`, the centred frame is written to
   `<run_dir>/plate-walls/<token>/<well>_<side>.tif` (`new_token()` once per
   run) by `encode_wall_frame` and `write_atomically`, its metadata the
   well, the side and the readback. An `OSError` there is not the
   registration's failure: that frame's `file` is `None`, and `assumed`
   gains, once, `<n> frames could not be saved: <the first error, on one
   line>`. Any other exception passes.
5. `ctx.progress(i, 4 × len(refs), "<well> <side>")`, `i` counting the sides
   done.

After a well's four sides: `fit_circle(points, floor_um=2 × pixel_size_um)`;
its `ValueError` is refused (`<well>'s walls do not fit a circle: <text>`);
then, each refused and naming the well and its frames' positions:

- `|size_error| > DIAMETER_TOLERANCE` (`plate_registration.algorithm`, 0.10):
  `<well> measures <D> µm across and the <plate> plate's wells <D0> µm: is it
  the right plate type?`;
- `rms_um > MAX_RMS_FRACTION × r`: `<well>'s wall points lie <rms> µm from a
  circle on average, more than <bound> µm: a frame may show something else
  than the wall (a bubble, a scratch, a label); look at its frames`.

**The plate**, once every well is measured: `calibrate(plate, refs=<the
fitted centres>, max_rotation_deg=params.max_rotation_deg)`; its
`ValueError` is refused with its text. Then each well's distance from the
fitted plate's centre for it (`well_center_um`); the worst above
`max_residual_um` is refused: `<well> is <n> µm from where the other wells
put it (bound <b> µm): record it again, or check the plate type`.

**The end.** `ctx.raise_if_stopped()`, then one move to the last well's
fitted centre: the operator sees on Live that the stage sits in the middle
of a well. Then the result: `calibrate`'s A1, rotation, residual and scale
error, `method="wall-images"`, `refs` and `walls` as §17.2. `assumed`, in this
order: the soft limits line (step 7); with two wells, `two wells: the
residual checks their distance only; a third well checks the fit`; the
saving line (step 4). The result is built once, never `model_copy`'d.

**On any exception during the moves** (a stop, a halt, a fault, a refusal of
the layer) the stage is left where it is: a WARNING names where
(`position_um()` read in its own `try`, `unknown` when that read fails too),
the well, the side and the start, and the exception propagates (FM-88,
FM-64). A `SafetyRefusedError` from a move (the soft limits, or a move
already under way) is refused as `<well>'s <side> wall: the move to (<x>,
<y>) µm was refused: <its text>`; `MicroscopeHaltedError` and
`MotionStoppedError` are not `SafetyRefusedError`s and pass unchanged. After
a refusal of the fit the stage stays at the last frame.

**Autofocus, turret, Z, light**: none is touched. The PFS stays as it is;
the tool reads no Z and moves none. The camera's exposure is the one set
(by Live, or by `smc snap`).

**Objective.** The tool works with the objective in place. On the Ti2-E,
10× is the one measured (§17.1), with a field of 1.55 mm; at 4× (a field of
about 3.9 mm) a recorded centre may be further off and a side takes fewer
frames, but no wall was imaged there yet (#198's next visit). On the 24-well
plate at 10×, `search_um` 3900 and a step of 1165 µm give 7 walk offsets: 2
frames per side when the recorded centre is within about 600 µm, 8 at most.

### 17.5 On the page (interface for #346)

What this fixes for the page, so that the page's issue can be planned
against it; the page's own design goes in `m5-web-ui.md` with that issue.

- `STAGES`' *Set up* gains `plate-walls` after `plate-registration`;
  `PAGE_PARAMS["plate-walls"] = ("plate", "refs")`, and `page_params(
  "plate-walls")` is `[f"plate={plate}", f"refs={json}"]` with the same
  `refs` captures as `plate-registration` (§15.3): one capture of a centre
  per well, a stage read, no move.
- Its next step, and the button that runs it, need at least two captured
  centres on the chosen plate; the card shows, per well, the centre, the
  diameter and the frames, then the residual and the rotation, from
  `walls`.

### 17.6 Tests

- `tests/unit/test_plate_walls.py` (`find_wall` and the rest on frames
  rendered by `PlateSample.render`, no stage):
  `test_a_wall_is_found_on_each_side_with_a_turned_mirrored_camera` (384 ×
  512, 30°, mirrored: FM-73); `test_the_centre_of_a_well_is_recovered_within_a_micrometre`
  (the cases of §17.1's table, with rough centres up to half a radius off);
  `test_a_frame_inside_a_well_has_no_wall`; `test_shading_without_a_wall_is_not_a_wall`
  (a ramp from 6000 to 9000); `test_a_frame_on_the_plastic_has_no_wall`;
  `test_the_neighbours_near_wall_is_not_taken`; `test_of_two_walls_in_a_frame_ours_is_taken`
  (48-well at 5 µm per pixel); `test_a_wrong_camera_turn_finds_no_wall`;
  `test_wall_points_agree_with_stage_of_pixel`; `test_walk_offsets_alternate_and_stay_within_the_search`;
  `test_fit_circle_recovers_a_circle_from_four_arcs`; `test_fit_circle_drops_an_outlying_arc`;
  `test_fit_circle_refuses_collinear_points`.
- **On the B3 frames**, in the same module, skipped (with the reason) when
  the frames are not there: they are private and not in a worktree. The
  folder is `SMC_SESSION2_DIR` when set, else
  `<checkout>/local/surveys/nikon-ti2/session2`.
  `test_b3s_centre_agrees_with_the_midpoint_of_its_walls`: §17.1's matrix
  and positions, the centre within 25 µm of (10 184.7, −10 091.4) µm (4.4
  measured), the diameter within 1 % of 15 300 µm (15 298 measured);
  `test_b3_with_an_upright_camera_finds_no_wall`. The develop session runs
  them once with the variable set and says so in its report.
- `tests/test_plate_walls_demo.py`, on `demo_microscope_with_sample` (the
  stage's `Velocity` at 100 000, as §6's test), `camera-calibration` run first
  on the same document (the seam, FM-124):
  `test_two_wells_register_the_plate_of_the_sample` (A1 within 2 µm, the
  rotation within 0.01°, of a 96-well sample turned 1.2° under a camera turned
  30° and mirrored, recorded centres 300 µm off);
  `test_three_wells_fill_the_plate_slot_with_their_walls` (`method`, `refs`,
  `walls`, 12 frames saved and named);
  `test_a_plate_of_another_pitch_is_refused_by_the_residual` (a sample of
  pitch 9.03 mm fitted as `96-well`); `test_a_recorded_centre_outside_its_well_is_refused_and_named`;
  `test_a_calibration_of_another_frame_is_refused_before_anything_moves`;
  `test_a_dry_run_is_refused`; `test_square_wells_are_refused`;
  `test_a_stop_leaves_the_stage_where_it_is`, `test_a_halt_leaves_the_stage_where_it_is`,
  `test_a_walk_outside_the_soft_limits_is_refused_before_anything_moves`
  (`limits_um` patched: the demo has none), `test_a_refused_move_names_the_well_and_the_side`
  (`move_to_um` raising `SafetyRefusedError` on its third call);
  `test_a_frame_that_cannot_be_saved_does_not_fail_the_registration`;
  `test_the_stage_ends_at_the_last_wells_centre`.
- `tests/unit/test_slots.py` gains the round trip of `WallFrame`, `WellFit`
  and `WallImages`, and a `PlateResult` written before #328 (no `walls`)
  still validating.

**Failure modes**: FM-32, FM-38, FM-50 (the B3 frames stay out of the
repository), FM-64, FM-73, FM-74, FM-75 (casts: frames are only read as
`float64`), FM-85, FM-88, FM-98, FM-121 (the default search obeys its
bound), FM-124, FM-125, FM-131, FM-142.

### 17.7 Not verified (for the owner)

- A registration at the Ti2-E: three wells, the residual, the time it
  takes, and `goto-well` to a fourth well. The check: record the fourth
  well where `goto-well` left the stage and run `plate-walls` on it with one
  of the first three; its `center_xy_um − recorded_xy_um` is how far
  `goto-well` landed from its centre, to be under 100 µm.
- **Which way the plate's columns run on the stage.** If the Ti2-E's stage
  puts column 2 at −X from column 1 (#327's question), two wells fit a
  rotation near 180° and `calibrate` refuses it as beyond `max_rotation_deg`.
  That is the safe answer, but not a registration: what the layer then does
  is #327's.
- `max_residual_um = 100`: no plate was registered from images yet.
- 4×, the light and exposure the walls need, other plates than the 24-well,
  and whether the PFS holds its lock while the stage looks at the walls.

## 18. The plate in another objective's frame (#340)

A plate registered with the 4× or the 10× is scanned with the 40×. The
offsets between objectives are the profile's, handed out by
`ObjectiveTurret.offset()` (m2 §13); this section records which objective a
plate was registered with and moves the plate into the frame of the
objective that uses it. Only XY moves: the layer applies no Z offset (m2
§13.2).

### 18.1 The plate slot records its objective — `slots/plate.py`

`PlateResult` (§3.2) gains, after `fit` (and after `walls`, §17.2, once
#328 has added it):

```python
objective: str | None = None
```

The turret label of the objective the plate's positions were taken with;
`None` when the stand has no turret, and in every document written before
#340 (unknown, not the one in place). It is a field with a default so that
older run documents still load.

It is the plate's own objective, for every method. `walls.objective`
(§17.2) stays as #328 defines it: the objective the wall frames were
taken with, which for `wall-images` is the same label.

### 18.2 Moving a plate between objectives — `smc/plugins/objectives.py` (new)

```python
def objective_for(
    stage: str, requested: str | None, turret: ObjectiveTurret | None
) -> tuple[str | None, bool]: ...


def objective_shift_um(
    origin: ObjectiveOffset, target: ObjectiveOffset
) -> tuple[float, float] | None: ...


@dataclass(frozen=True, slots=True)
class PlateInFrame:
    plate: PlateResult
    objective: str | None
    shift_um: tuple[float, float] | None
    assumed: tuple[str, ...]


def plate_for_objective(
    plate: PlateResult, objective: str | None, turret: ObjectiveTurret | None
) -> PlateInFrame: ...
```

- **`objective_for`** is `scan-plan`'s `_objective_for` (§7.3) moved here
  unchanged, messages included, with the stage's name as its first
  argument: the requested objective, checked against the turret, or the one
  in place; an empty position or a requested objective on a stand with no
  turret is a `PluginRunError` of `stage`. `scan-plan` imports it.
- **`objective_shift_um(origin, target)`** is `target.xy_um −
  origin.xy_um`, per axis; `None` when either is `None` or the two
  references differ.
- **`plate_for_objective`**, in this order:
  1. `objective == plate.objective` (both `None` included): `plate`
     itself, `shift_um=(0.0, 0.0)`, no line.
  2. `plate.objective is None`: `plate` itself, `shift_um=None`, the line
     `the plate does not record the objective it was registered with: no
     offset between objectives is applied for <objective>`.
  3. `objective is None`: `plate` itself, `shift_um=None`, the line `the
     objective in use is unknown: the plate's positions are
     <plate.objective>'s, and no offset between objectives is applied`.
  4. `turret is None`, or `objective_shift_um(turret.offset(plate.objective),
     turret.offset(objective))` is `None`: `plate` itself, `shift_um=None`,
     the line `the offset between <plate.objective> and <objective> is not
     measured (the profile's [objectives]): the positions are
     <plate.objective>'s, not corrected`.
  5. Otherwise the plate **translated**: a new `PlateResult`, built once by
     `PlateResult.model_validate` from the dump (never `model_copy`, which
     skips validation), with `a1_center_xy_um` and every `refs` entry's
     `x_um`, `y_um` plus `shift_um`, `objective=objective`, everything else
     as it was (the rotation is unchanged: the shift is a translation, m2
     §13.2). No line: the offset is measured. The plugin logs at INFO the
     shift and both offsets' `measured`.

  The line of 2, 3 or 4 is also logged at WARNING by the plugin that
  receives it. `plate.assumed` is left as it is: the caller copies it as it
  does today.

Translating `a1_center_xy_um` moves every well centre and wall point by
exactly the shift (measured: 0.0 µm of difference on A1, D7 and H12 of a
96-well plate at 3°, centres and left walls, useq-schema 0.9.2).

### 18.3 Who fills it, who uses it

- **`plate-registration`** (§5.3): `PlateRegistrationParams` gains
  `objective: str | None = Field(None, description="The objective the
  references were centred with, by its turret label; default the one in
  place when the registration runs.")`. `run` calls
  `objective_for(_NAME, params.objective, microscope.get(ObjectiveTurret))`
  first, before the fit, and the result carries it. A turret read that
  raises is not caught (the runner records the stage `failed`).
- **`plate-finding`** (§6): `objective_for(_NAME, None, turret)` at the
  start of every attempt, before anything moves. `SurveyCheckpoint` gains
  `objective: str | None = None`, written by the attempt that creates the
  checkpoint. A resumed attempt whose objective differs from a checkpoint's
  that is not `None` is refused before anything moves: `the survey began
  with <a> and the objective in place is <b>: turn the turret back to <a>
  and resume, or run the survey again without --resume`. After the last
  point and before the fit, the objective is read again; a different one
  ends the attempt with `the objective changed during the survey, from <a>
  to <b>: run the survey again without --resume`. The result carries it.
- **`plate-walls`** (§17.4, #328): its result sets `objective` to
  `WallImages.objective`, the objective in place. Of #328 and #340, the
  issue developed second adds that line and its test
  (`test_the_wall_registration_records_its_objective`); if #340 lands
  first, #328 reads the objective with `objective_for(_NAME, None, turret)`,
  so that an empty position is refused as everywhere else.
- **`goto-well`** (§5.3): `objective_for(_NAME, None, turret)`, then
  `plate_for_objective(plate, objective, turret)` before the target is
  computed; the target is the frame's plate's centre or wall point.
  `GotoWellResult` gains `objective: str | None`, `plate_objective: str |
  None` and `shift_um: tuple[float, float] | None`; the frame's line joins
  `assumed` after the plate's.
- **`scan-plan`** (§7.3): after `objective_for`, `plate_for_objective(plate,
  objective, turret)`; every later use of the plate in `run` (the grid's
  well size, `plan_fields`) takes the frame's plate, so `plate_plan` and
  every field are in the plan's objective's frame. That objective is the
  requested one when it is not in place: the plan is for it, and `scan`
  refuses to run it with another (§8). `ScanPlanResult` (§3.4) gains
  `plate_objective: str | None = None` and `objective_shift_um: tuple[float,
  float] | None = None` (defaults for older documents: unknown). The
  frame's line joins `assumed` before the basis's.
- **`scan`** (§8): unchanged. It visits the plan's fields, which are in the
  plan's objective's frame, and already refuses another objective in place
  at the start and at both ends of every well.
- **Downstream** (`cells`, `density`, `route`, `report`, `nis-export`, the
  page's maps of the scan) reads wells from `plan.plate_plan`, now in the
  scan's frame, and is unchanged. `route` to another objective keeps its
  `assumed` line `no offset between the two objectives is applied`: whether
  NIS-Elements applies its own is not known (m2 §13.8), and applying ours
  is a later issue.
- **The plate page** is unchanged: it registers through `plate-registration`
  with the objective in place, and goes to a well through `goto-well`. Its
  map of the plate draws the registration's frame; a shift of tens of µm
  does not show at the scale of a plate.

### 18.4 Tests

- `tests/unit/test_objective_offsets.py` (pure, a stub turret):
  `objective_shift_um`'s sign (target minus origin, with neither offset
  zero), unknown on either side, differing references; each rule of
  `plate_for_objective` with its exact line; the translated plate's A1,
  every ref, `objective`, rotation and fields unchanged, and a refusal of
  the translated values by validation not bypassed.
- `tests/test_goto_well_demo.py`: on a demo profile whose `[objectives]`
  names `Nikon 20X Plan Fluor ELWD` as the reference, with the 10× at
  `(5.0, -2.5)` and the 40× at `(-32.5, 9.75)` (a shift 10× → 40× of
  `(-37.5, 12.25)`, anisotropic, FM-73), a plate registered with the 10×
  and the 40× turned in: the target is the registration's plus the shift,
  written from the numbers above and not from the code (FM-124); the same
  objective: no shift, no line; an offset unknown: the target unshifted and
  the line.
- `tests/test_scan_plan_demo.py`: a plate registered with the 10× planned
  for the 40× (requested, not turned in): every field and every well of
  `plate_plan` is the 10× plan's plus the shift; `plate_objective` and
  `objective_shift_um` are recorded; a plate with no objective gives the
  line of rule 2.
- `tests/test_plate_registration_demo.py`, `tests/test_plate_finding_demo.py`:
  the result records the objective in place and the one requested; an
  objective not on the turret is refused; the survey's resume with another
  objective and a turn during the survey are refused.
- `tests/unit/test_slots.py`: a `PlateResult` dump without `objective`
  loads with `None`; the same for `ScanPlanResult`'s two fields.

### 18.5 Not verified

- m2 §13.8. A plate registered at 4× and scanned at 40× at the stand is the
  first real test of the whole.

## 19. The sides of a wall, in the frame they are named in (#327)

A wall is named by a direction, and a direction means something in one frame
only. Three frames meet at a plate: the **image** the operator looks at (`left`,
`right`, `top`, `bottom`), the **stage** (`+X`, `-X`, `+Y`, `-Y`: the X and Y of
`smc stage get`, M1 §2) and the **plate**, the stage's turned by the plate's small
rotation. Until #327 one set of words, `left right top bottom`, was typed in the
image frame by the operator (on the page, and in the docs' steps) and read in the
stage frame by the layer (`centre_from_walls`, `wall_position`, `goto-well`).
FM-189 is that failure. This section fixes which words each part uses; the page's
side of it is m5 §13.

### 19.1 What was measured (2026-11-09, planning #327)

On throwaway prototypes that are not committed; main at `13687da`, Python 3.13,
useq-schema 0.9.2.

- **The incident, reproduced.** `centre_from_walls("24-well", "B3", …)` with the
  two touches of Ti2-E session 2 as the operator named them on the image, `left` at
  (17967.9, −10237.1) and `bottom` at (10368.3, −2334.1), gives (25767.9, 5465.9),
  both axes assumed: the issue's seed, 15.5 mm from B3's centre of §17.1,
  (10182.0, −10087.9).
- **A pair named on the image is refused, not misread.** The four touches of
  §17.1's table, named as the files are (`left` at X 18067.4, `right` at X 2468.0),
  are refused with `B3's left and right walls are -15599 µm apart, a 24-well well is
  15600 µm (-200.0%): a touch of another well, or off the wall`. The signed size
  catches a swapped pair, and names a wrong cause for it.
- **The camera's turn decides which side is which.** From the matrix of §3.1 for
  each camera, the stage side of each image side, and what a stage step does to the
  image (the content moves by `−M⁻¹ · d`, in pixels, `+col` right and `+row` down):

  | camera | `left` | `right` | `top` | `bottom` | +X step: the image moves | +Y step: the image moves |
  |---|---|---|---|---|---|---|
  | 0° | −X | +X | +Y | −Y | left | down |
  | 0°, mirrored | +X | −X | +Y | −Y | right | down |
  | 90° | −Y | +Y | −X | +X | up | left |
  | 90°, mirrored | +Y | −Y | −X | +X | up | right |
  | 180° (the Ti2-E at 10×) | +X | −X | −Y | +Y | right | up |
  | 180°, mirrored | −X | +X | −Y | +Y | left | up |
  | −90° | +Y | −Y | +X | −X | down | right |
  | −90°, mirrored | −Y | +Y | +X | −X | down | left |

  The words `left right top bottom` that the layer read until now are the first row
  only: right on an upright camera, reversed on both axes at 180°. The 180° row is
  Ti2-E session 2 as the frames and Drive showed it (§17.1: the image's right is −X
  and its bottom +Y; `→` moved the image right and `↑` up); `camera-calibration` was
  not run there. A real `camera-calibration` on the synthetic stand (384 × 512, the
  camera at 0°, 90°, 180°, 180° mirrored, 3° and 30° mirrored) measured the rotation
  within 0.02° and the mirror right, and gave the sides of the rows above; at 30° no
  side is on a stage axis (30.0° off).

### 19.2 One frame for each word

- **The layer takes stage directions only.** `WallTouch.side` is `WallSide`
  (`"+X"`, `"+Y"`, `"-X"`, `"-Y"`, §17.2's), and `GotoWellParams.side` is
  `Literal["centre", "+X", "-X", "+Y", "-Y"]`. A side is the wall reached from the
  well's centre by going that way **along the plate's own axes** (the stage's,
  turned by the plate's rotation): `-X` is what `left` was, `+X` `right`, `-Y`
  `bottom` and `+Y` `top`. `wall_position(result, well, side)` takes the same
  four; `centre_from_walls` reads them per axis as §5.2 says, with the names
  changed, and `plate-walls` already used them.
- **The old words are refused, never read in the other frame.** One helper in
  `slots/plate.py`, `refuse_image_side(value)`, used as a `field_validator(...,
  mode="before")` on `WallTouch.side` and on `GotoWellParams.side`: a value in
  `IMAGE_SIDES` raises `ValueError` with the text below; any other value goes on
  to the `Literal`'s own check.

  ```text
  side 'left' is a side on the image: a wall is named by a stage direction, -X, +X, -Y or +Y (the plate page names walls on the image and maps them with the camera calibration)
  ```

  A run document written before holds the old words in `params.walls`; they are
  not translated (a translation is a guess of the frame, the defect itself): `smc
  run` of such parameters is refused with the text above, and the page reads no
  capture its model refuses (`_captured`, logged, unchanged). The only such
  documents are Ti2-E session 2's (smc 0.0.1).
- **Why not keep the words and add image ones.** The same word in two frames is
  the defect; a renamed word is refused where it is used in the old sense, and
  cannot be read in the other frame by anyone.
- **`ImageSide`, `IMAGE_SIDES`** live in `slots/plate.py` beside `WallSide`
  (`Literal["left", "right", "top", "bottom"]` and the tuple of its four values),
  exported by `smc.plugins.slots`.
- **A swapped pair says so.** `_check_size`'s signed size is kept. When the size
  is negative and its magnitude is within `DIAMETER_TOLERANCE` of the nominal, the
  refusal is, with `lo`, `hi` the pair's sides (`-X`, `+X` or `-Y`, `+Y`) and
  `gap` the magnitude in µm:

  ```text
  B3's -X and +X walls are swapped: the +X wall is 15599 µm on the -X side of the -X wall, a 24-well well's width; touch them again, each named by its direction from the well's centre
  ```

  (`height` for the Y pair.) Any other size off by more than the tolerance keeps
  today's text with the sides named in the new words (`B3's -X and +X walls are
  9200 µm apart, a 24-well well is 15600 µm (-41.0%): a touch of another well, or
  off the wall; touch them again`). Messages of the other refusals: `no X wall for
  B3: give -X and/or +X`, `no Y wall for B3: give -Y and/or +Y`, `B3: the -X wall
  is given twice`.

### 19.3 Image sides — `smc/plugins/frames.py` (new)

Pure arithmetic on the camera matrix of §3.1: it imports `Matrix2` from
`geometry` and the side types from `slots/plate.py`, and nothing from
`smc.hardware`. It is the only place that turns an image word into a stage word.

```python
ImageDirection = Literal["left", "right", "up", "down"]
QUARTER_TURN_TOLERANCE_DEG = 5.0


def stage_side_of_image_side(side: ImageSide, matrix: Matrix2) -> WallSide: ...
def image_direction_of_stage_step(
    step: WallSide, matrix: Matrix2
) -> ImageDirection: ...
```

- **An image side** is a direction from the well's centre on the image: `right` is
  the matrix's first column `(m00, m10)` (a pixel to the right of the frame's
  centre is `M @ (1, 0)` µm from it), `bottom` its second `(m01, m11)`, and `left`
  and `top` their opposites. The stage side is the axis on which that vector has
  the larger component, with the component's sign.
- **A stage step** moves the image content by `−M⁻¹ @ d` pixels for a stage step
  `d` (`+X` is `(1, 0)`, `-Y` is `(0, -1)`); the words are `right` for `+col`,
  `left`, `down` for `+row` and `up`, again on the axis of the larger component.
- **The quarter turn.** Both functions refuse a vector more than
  `QUARTER_TURN_TOLERANCE_DEG` off a stage axis, `off = degrees(atan2(minor,
  major))` with `major`, `minor` the larger and smaller component's magnitudes,
  written `not off <= tolerance` so that a `nan` is refused (FM-104). 5° is chosen,
  not measured: a wall touched `t` µm off the well's mid-line errs by `t · sin(5°)`
  on the axis read, 87 µm for 1 mm, about what a hand-touched wall is worth at 10×
  (§5.2 gives the plate's analogue), and a mounted camera is within a degree or two
  of a quarter turn. A camera further turned is for centring wells (`refs`) or for
  `plate-walls`, which takes the matrix itself.
- **The refusals** are `ValueError`s with these texts: a matrix that cannot be
  inverted or is not finite, `the camera matrix cannot be inverted (determinant
  <d>); run camera-calibration`; a vector off a quarter turn, `the camera is turned
  <off:.1f>° away from the stage's axes, a quarter turn at most 5° off: an arrow
  moves the image diagonally and an image side is not a stage side`.
- **Which matrix.** Rotation and mirror belong to the optical path (the camera's
  port, the mirrors), not to the objective, so the page uses the latest *usable*
  `camera-calibration` whatever its objective (m5 §13.1). A calibration is of the
  path it was measured in: after a change of camera port or light path,
  `camera-calibration` is run again (said in the plugin's docs).

### 19.4 Who maps

The plugins never see an image side. `plate-registration`, `goto-well` and
`plate-walls` take stage directions, so each stays a pure function of its
parameters and the `plate` slot: a plugin that read a camera calibration would
make a registration depend on a slot that its parameters do not show (M3 §1).
The plate page, whose operator reads the image, names walls on the image and maps
them with `frames.py` before it builds a parameter (m5 §13). On the command line
the operator names stage directions, and the docs say which frame each word is in
and how to read the stage's from the camera calibration (the table of §19.1).

### 19.5 Tests (FM-189's check)

- `tests/unit/test_frames.py` (new): the eight rows of §19.1, **written out in
  the test** and not computed with the code (FM-124), for each of the four image
  sides and both steps; `((-1.1, 0.0), (0.0, 1.1))` is the Ti2-E's matrix of §17.1;
  a camera 4.0° off a quarter turn is accepted and one 6.0° off refused (the
  bound is 5.0°, so neither value sits on it) with the text above; a `nan` and a
  singular matrix are refused.
- `tests/unit/test_plate_registration.py` and `tests/test_plate_registration_demo.py`:
  the incident's two touches, named in stage directions as the image-frame table
  gives them (`left` → `+X` at (17967.9, −10237.1), `bottom` → `+Y` at (10368.3,
  −2334.1)), give B3's centre at (10167.9, −10134.1), 14 and 46 µm from the
  measured centre, not (25767.9, 5465.9); the four touches of §17.1 give
  (10267.7, −10038.05) with a width of 15599.4 µm and a height of 15400.9 µm; the
  old words are refused with §19.2's text, on `WallTouch` and on `GotoWellParams`;
  a swapped pair gets the swapped text and an off-size pair the other.
- `tests/test_goto_well_demo.py`: `side=+X` and `side=-Y` go to the points that
  `left` and `bottom` went to; the demo's wall points are unchanged by the rename.
- The page's tests (m5 §13.3) hold the end-to-end check: a stand whose matrix is
  the Ti2-E's, walls captured under image names, the centre found where the walls
  are.

### 19.6 Not here

- **The fitted centre as a cross on Live** before the registration is kept (the
  issue's third question): it needs an overlay on the live frame (m5 §5), a design
  of its own. This issue's guards are the mapping, the swapped-pair refusal and the
  page's note of the mapping; the owner decides whether the cross follows.
- **A profile key for the camera's orientation.** It would be a second source for
  what `camera-calibration` measures; the page asks for the calibration instead.
- **A mode of the page that needs no calibration** (an opposite pair oriented from
  the two positions alone): a second mechanism for a case that one
  `camera-calibration` removes.

### 19.7 Not verified (for the owner)

- **Which way the plate's grid runs on the Ti2-E's stage** (#327's question, #334).
  The check, at 10× with B3 registered from its walls (a seed, rotation 0 assumed):
  `smc run goto-well --param well=C4`. `useq` puts C4 at (+19 000, −19 000) µm from
  B3 (24-well pitch, measured 2026-11-09), so from B3's centre of §17.1 at about
  (29 182, −29 088). A well under the objective within a couple of mm means the
  columns run towards +X and the rows towards −Y, as `useq`'s: nothing to do. Else
  centre C4 and read `smc stage get`, and write both positions in the report. Two
  wells whose columns run towards −X fit a rotation of 180° and `calibrate`
  refuses it as beyond `max_rotation_deg` (measured: `the fit turns the plate by
  180.0°, beyond ±10°: a reference well is misidentified …`), the safe answer; what
  the layer does for a plate that lies turned on the stage is a design after the
  measurement, not a guess now.
- **`camera-calibration` at 10× on the Ti2-E**: `rotation_deg` near ±180° and
  `mirrored` false are the 180° row of §19.1. Any other result is the finding.
- **Which way NIS-Elements' own XY control moves the image** (press `→` and `↑`,
  then the joystick the same ways, and note each): the page's note under the pad
  says what the arrows do here, and the owner compares.

## 20. The offsets between objectives, measured from the plate (#380)

The owner (2026-11-09): the plate is registered with the 4×, and the offsets
4× → 10× and 4× → 40× should come from the plate itself, not from a stage
micrometer centred by hand (m2 §13.6). `smc/plugins/objective_offsets/`,
entry point `objective-offsets =
"smc.plugins.objective_offsets.plugin:ObjectiveOffsetsPlugin"`, slot
`objective-offsets`. The tool reads `plate` (a registration by `plate-walls`,
§17), turns the turret to each target objective, re-measures under it the
centres of the wells the registration measured, with `plate-walls`' own loop
and the target's own camera calibration, and gives each target's offset from
the registration's objective. `smc objective offset --from-run-doc` shows what
that puts into the profile's `[objectives]` (m2 §13.3), and `--write` writes it.

**What the operator does** (the Ti2-E, a 24-well plate):

1. Once per objective, in a run document of its own, which later sessions
   reuse: `smc objective set "2-Plan Fluor 10x" --confirm`, the stage on a
   well's wall, `smc run camera-calibration -p nikon-ti2 --run-doc
   cal-10x.json`; the same with the 40× into `cal-40x.json`.
2. The 4× in place: `camera-calibration` and `plate-walls` into `run.json`
   (§17).
3. `smc run objective-offsets -p nikon-ti2 --run-doc run.json --param
   'targets=["2-Plan Fluor 10x", "3-Plan Apo LmbdD0.25 40x"]' --param
   'calibrations=["cal-10x.json", "cal-40x.json"]' --param confirm_turns=true`.
4. `smc objective offset -p nikon-ti2 --from-run-doc run.json --session
   "session 3 (#NNN)"` prints what it would write; the same with `--write`
   writes it; `smc objective offset -p nikon-ti2` reads it back against the
   turret.

**The same feature is the well's measured centre.** m2 §13.2 compares the
stage positions that centre one feature under two objectives. Here the
feature is a well's centre as `plate-walls` measures it from its four walls
(a point, which one wall is not): under the registration's objective it is
`plate.walls.wells[i].center_xy_um`, under the target the same loop gives it
again. The plate's fit is not used: its prediction for a well carries the
fit's residual (up to `max_residual_um`, §17.4), which the two measured
centres of one well do not.

**A calibration per target, from another run document.** The camera's turn
and mirror decide which way every wall lies (§17.1), and the scale changes
with the objective: a wall is looked for with the target's own matrix or not
at all. A `camera-calibration` run in `run.json` would replace the one the
plate was registered with and make `plate` stale (§1.7), so each target's
calibration comes from a document of its own, loaded by the rule
`camera-calibration` reuses one with (§4.2, `from_run_doc`).

**Not here**: the plate page (its next step after a registration, the
objective in place at each step and the turns' confirmation; a follow-up
issue, after #373); a Z offset (m2 §13.2: the layer applies none; a focused Z
needs the operator, which a plugin cannot ask, §5.1, or a declared PFS lock
of #349); a plate registered by `plate-registration` or `plate-finding` (their
positions are centred by eye or fitted, not measured centres); turning back to
the registration's objective at the end (FM-88); the magnifier lever (FM-103:
the offsets are keyed by objective alone, and the result says so).

### 20.1 What was measured (2026-11-09, planning #380)

On throwaway prototypes that are not committed; main at `17d72c7`,
pymmcore-plus 0.18.1, useq-schema 0.9.2, numpy 2.5.3, pydantic 2.13.5,
scikit-image 0.26.0, Python 3.13.0.

- **The demo-sample stand** (`profiles/demo-sample.toml`): the turret's labels
  `Nikon 40X Plan Fluor ELWD`, `Nikon 10X S Fluor` (in place at open),
  `Objective-2`, `Nikon 20X Plan Fluor ELWD`, `Objective-4`, `Objective-5`;
  frames of 512 × 512; the camera reports 1.0, 0.5 and 0.25 µm per pixel at
  10X, 20X and 40X (MMCore's `Res10x`, `Res20x`, `Res40x`, which win over the
  profile's sizes, m1 §6); the XY stage reports no soft limits; Z reads 0.0
  before and after a turn; a turn returns in 0.11 s. `select` without
  `confirm` is a `SafetyRefusedError`: `turning the objective turret can drive
  an objective into a loaded plate (objective_turret 'Objective') — pass
  confirm=True once the stage is clear`.
- **A shift per objective, recovered.** The sample camera patched to render
  at the stage position minus a shift per objective in place (10X `(5.0,
  −2.5)`, 20X `(0, 0)`, 40X `(−32.5, 9.75)`), the camera turned 30° and
  mirrored, the plate turned 1.2°:
  - `camera-calibration` at each objective, each into its own document:
    0.99967, 0.4995 and 0.2498 µm per pixel, 29.99°, 29.97° and 29.93°,
    mirrored each time.
  - `plate-walls` at 10X on A1, H12 and A12, recorded 300 µm off: 30 frames
    in 1.0 s, each centre within 0.013 µm of the sample's (shifted by the
    10X's).
  - Under 20X and under 40X, §17.4's loop (`_measure_well`) with the
    target's calibration and the 10X centres as the recorded ones: 24 frames
    for the three wells (two per side) in 0.8 s each. The mean of the three
    wells' differences was `(−4.9961, 2.4944)` µm for the declared `(−5.0,
    2.5)` and `(−37.4941, 12.2530)` for `(−37.5, 12.25)`: within 0.009 µm.
    The largest distance between two wells' differences was 0.012 and 0.004
    µm. With a search of 1000 µm the 40X walk has 21 offsets (33 at half the
    radius, plate-walls' default).
- **Parameters.** `parse_params` takes `tuple[str, ...]` and `tuple[Path,
  ...]` as JSON lists (a Windows path's backslashes doubled inside the JSON);
  `targets=a,b` is refused with `'targets' takes JSON, for example ["A1",
  "B2"]`, and two missing required fields are named in one error.
- **Test doubles.** `Microscope.override(ZStage, w)` refuses a delegate that
  forwards through `__getattr__` (`TypeError`: the protocol check sees no
  method); with each method written out it passes, and the turret's own wait
  for Z (m2 §8.11 D2) reads it (11 reads during a demo turn). A turret
  double whose `select` calls the real one and then sets the Z double
  drifting (1 µm per read) gives the layer a still Z and the plugin a moving
  one, with no timing in the test. A `plate` record rewritten with
  `write_stage(path, StageRecord.model_validate({**record.model_dump(
  mode="json"), "result": changed}))` stays usable (`unusable` is `None`) and
  validates as a `PlateResult`. `PlateSample.render` replaced at class level
  by a wrapper that records `(x_um, y_um)` and calls the original sees every
  snap of the demo-sample camera. Two snaps at one place are **not** equal
  (the noise is drawn per snap), and the demo stage reads `(123.495,
  −45.255)` after a move to `(123.5, −45.25)`: a test compares with the
  readback, never with the target (FM-142).
- **tomli-w 1.0.0** (the floor of `pyproject.toml`) writes `{"objectives":
  ...}` as an `[objectives]` table and one `[objectives.offsets."<label>"]`
  table per entry, each array one number per line, a label with `"` and `\`
  escaped, floats as their `repr`; `tomllib` reads it back equal.

### 20.2 `plate-walls`' loop, for two tools — `plate_walls/loop.py` (new)

`plate-walls`' hardware loop and its checks before anything moves move out of
`plate_walls/plugin.py` into `plate_walls/loop.py`, public within the package
and unchanged in behaviour: `plate-walls`' refusals, warnings, progress and
files stay character for character what they are, and its tests, untouched,
prove the move. Every message that named `plate-walls` takes the stage's name
instead (`WallWalk.stage`, or the `stage` argument).

```python
@dataclass(frozen=True, slots=True)
class WallWalk:
    stage: str
    plate: str
    radius_um: float
    search_um: float
    step_um: float
    matrix: Matrix2
    shape: tuple[int, int]
    pixel_size_um: float
    settle_s: float
    folder: Path | None
    folder_text: str | None
    no_wall_hint: str
```

```python
class Where:
    __slots__ = ("text",)


def check_calibration(
    stage: str,
    calibration: CameraCalibration,
    objective: str | None,
    has_turret: bool,
    shape: tuple[int, int],
) -> None: ...
def check_frame(
    stage: str, matrix: Matrix2, shape: tuple[int, int], diameter_um: float
) -> None: ...
def checked_search(
    stage: str, search_um: float | None, radius_um: float, *, default_um: float
) -> float: ...
def check_walk_size(
    stage: str, n_wells: int, search_um: float, step_um: float
) -> None: ...
def check_soft_limits(
    stage: str, xy: XYStage, refs: tuple[WellRef, ...], offsets: tuple[float, ...]
) -> list[str]: ...
def move(
    stage: str, xy: XYStage, x_um: float, y_um: float, what: str
) -> tuple[float, float]: ...
def measure_well(
    xy: XYStage,
    camera: Camera,
    ctx: RunContext,
    walk: WallWalk,
    ref: WellRef,
    offsets: tuple[float, ...],
    where: Where,
    save_errors: list[str],
    on_side: Callable[[str], None],
) -> tuple[WellFit, int]: ...
def warn_left(
    stage: str, ctx: RunContext, xy: XYStage, start: XY, where: Where
) -> None: ...
```

- `WallWalk` is today's `_Walk` with two fields: `stage`, the plugin whose
  refusals these are, and `no_wall_hint`, the end of the "no wall found"
  refusal after its first clause, with `{well}` where the well's name goes
  (`str.format(well=...)`). `plate-walls`' is today's text, `check that the
  light is on (the inside of a well is brighter than the plastic in
  brightfield) and that the stage was in {well} when it was recorded, then
  record it again`.
- `checked_search` takes the default from its caller (`plate-walls`:
  `DEFAULT_SEARCH_FRACTION × r`, as today) and keeps the bound: a value above
  the radius is refused, the default included.
- `measure_well` no longer counts progress: after each side it calls
  `on_side("<well> <side>")`. `plate-walls` passes a function that counts the
  sides done and calls `ctx.progress(i, 4 × len(refs), text)`, which is
  today's call. `_snap`, `_measure_side` and `_save` move as private helpers
  of `loop.py`; `_save` writes `<folder>/<well>_<side>.tif` as today.
- `warn_left`'s two texts start with the stage's name where they say
  `plate-walls` today.
- Left in `plugin.py`: `PlateWallsParams`, `PlateWallsPlugin.run`,
  `_checked_refs`, `_fit_plate` and the constants of the refusals that only
  `plate-walls` makes.

### 20.3 A calibration from another run document — `camera_calibration/plugin.py`

```python
def calibration_from(
    stage: str, path: Path, *, profile: str, dry_run: bool
) -> tuple[CameraCalibration, str]: ...
```

The first half of today's `_reuse`, public: the document loaded, the rule
`RunDocument.unusable("camera-calibration", profile=..., dry_run=...)`, the
record's result validated, each refusal a `PluginRunError` of `stage` with
`_reuse`'s texts. It returns the calibration and when it was measured (the
record's `finished_utc`, ISO to the second, or `at an unknown time`). `_reuse`
calls it with `camera-calibration`, then checks the objective and the frame
and builds the reused result as today; its texts do not change.

### 20.4 The slot — `slots/objective_offsets.py` (slot `objective-offsets`)

```python
class OffsetWell(SlotValue):
    well: str
    origin_xy_um: tuple[float, float]
    delta_xy_um: tuple[float, float]
    fit: WellFit
```

```python
class TargetMeasure(SlotValue):
    objective: str
    calibration: str
    calibration_measured: str
    pixel_size_um: float = Field(gt=0.0)
    z_after_turn_um: float | None
    wells: tuple[OffsetWell, ...]
    shift_xy_um: tuple[float, float]
    spread_um: float = Field(ge=0.0)
    max_spread_um: float = Field(gt=0.0)
    frames: int = Field(ge=0)
    folder: str | None
```

```python
class MeasuredOffset(SlotValue):
    objective: str
    xy_offset_um: tuple[float, float]
```

```python
class ObjectiveOffsetsResult(PluginResult):
    plate_objective: str
    reference: str
    reference_in_profile: bool
    measures: tuple[TargetMeasure, ...]
    offsets: tuple[MeasuredOffset, ...]
    objective_in_place: str
    elapsed_s: float = Field(ge=0.0)
```

- **Every position is a stage position** (`XYStage.position_um()`, M1 §2),
  never an image side (FM-189).
- `OffsetWell`: `origin_xy_um` is the centre `plate-walls` measured with the
  registration's objective (`plate.walls`); `fit` is the well under the
  target, as `plate-walls` records one (its `recorded_xy_um` is
  `origin_xy_um`); `delta_xy_um` is `fit.center_xy_um − origin_xy_um`.
- `TargetMeasure`: `calibration` is the run document given for this target,
  as typed (`as_posix()`, never resolved: FM-24, FM-50) and
  `calibration_measured` when it was measured (§20.3); `pixel_size_um` is that
  calibration's; `z_after_turn_um` is Z where the stand left it after the turn
  (§20.6), `None` on a stand with no focus drive; `wells` in the order run;
  `shift_xy_um` the mean of the wells' `delta_xy_um`, the target's offset from
  the registration's objective; `spread_um` the largest distance between two
  wells' `delta_xy_um`; `max_spread_um` the bound it was held to; `frames` the
  frames taken under this target; `folder` run-relative
  (`objective-offsets/<token>/<n>`, `n` the target's place in `targets` from
  1; no label in a path) or `None` when frames are not saved.
- `ObjectiveOffsetsResult`: `plate_objective` is `plate.objective`;
  `reference` the objective the offsets are from, the profile's (`True` in
  `reference_in_profile`) or, when it names none, `plate_objective`
  (`False`); `measures` in the order of `targets`; `offsets` what `--write`
  writes (§20.5's `offsets_from_reference`): every objective the run measured
  or registered with, except `reference`, the registration's objective first
  and then the targets in their order; `objective_in_place` the last target,
  where the turret is left.
- `slots/__init__.py` exports the four models.

### 20.5 Algorithm — `objective_offsets/algorithm.py`

Pure: no device, no file, no numpy needed.

```python
#: The wells may disagree by this many pixels of the coarser of the two
#: objectives (m2 §13.6: 5.5 µm at the Ti2-E's 4x). Chosen, not measured.
SPREAD_PIXELS = 2.0
#: At most this many targets in one run, so that the walks' bound holds in all
#: (FM-140). Chosen: a turret holds six positions.
MAX_TARGETS = 5


def mean_shift_um(deltas: Sequence[tuple[float, float]]) -> tuple[float, float]: ...
def spread_um(deltas: Sequence[tuple[float, float]]) -> float: ...
def default_max_spread_um(origin_pixel_um: float, target_pixel_um: float) -> float: ...
def offsets_from_reference(
    origin: str,
    shifts: Sequence[tuple[str, tuple[float, float]]],
    reference: str | None,
) -> tuple[str, tuple[tuple[str, tuple[float, float]], ...]]: ...
```

- `mean_shift_um` is the mean per axis; `ValueError` for no delta.
- `spread_um` is the largest `math.dist` between two deltas, `0.0` for one
  delta; `ValueError` for none.
- `default_max_spread_um` is `SPREAD_PIXELS × max(origin, target)`;
  `ValueError` for a size that is not finite and positive.
- `offsets_from_reference(origin, shifts, reference)`: `d(origin) = (0, 0)`
  and `d(t) = shift` for each target; `R` is `reference`, or `origin` when it
  is `None`; `ValueError` when `R` is not `origin` and not among the targets
  (the plugin refuses that before anything turns, §20.6). The result is `R`
  and, for `origin` then each target in order, leaving out `R`, the label and
  `d(label) − d(R)` per axis, each `+ 0.0` (no `-0.0`). With `R = origin`
  each target's offset is its shift; with `R` a target, the origin's offset
  is `−d(R)` and every other target's `d(t) − d(R)` (m2 §13.2: `p_O = p_R +
  offset(O)`).

### 20.6 Plugin — `objective_offsets/plugin.py`

```python
class ObjectiveOffsetsParams(PluginParams):
    targets: tuple[str, ...] = Field(min_length=1, max_length=MAX_TARGETS)
    calibrations: tuple[Path, ...] = Field(min_length=1, max_length=MAX_TARGETS)
    confirm_turns: bool = False
    wells: tuple[str, ...] = Field((), max_length=6)
    search_um: float | None = Field(None, gt=0.0, le=MAX_XY_OFFSET_UM)
    max_spread_um: float | None = Field(None, gt=0.0, le=100.0)
    settle_ms: float = Field(0.0, ge=0.0, le=10_000.0)
    save_frames: bool = True
```

`requires = (XYStage, Camera, ObjectiveTurret)`, `reads = ("plate",)`,
`reads_optional = ()`, `provides = "objective-offsets"`. Z is
`microscope.get(ZStage)`, `None` without one. No checkpoint: a run takes about
a minute, and a stopped one is run again. Each field has a `description`:

- `targets`: the objectives to measure, by turret label, in the order the
  turret turns to them (JSON: `targets=["2-Plan Fluor 10x"]`); the last one is
  left in place.
- `calibrations`: one run document per target holding a `camera-calibration`
  done with that target in place (JSON list of paths, as typed; relative to
  the working directory, as `from_run_doc`).
- `confirm_turns`: the stage is clear for the turret to turn to each target
  with the plate on it (FM-14).
- `wells`: wells the registration measured, at least two; default all of
  them.
- `search_um`: how far each wall may lie from where the registration's centre
  puts it, in µm; default `min(MAX_XY_OFFSET_UM, r)`; never above the radius.
  An offset larger than the profile's bound could not be written anyway.
- `max_spread_um`: how far apart the wells' differences may lie; default
  `SPREAD_PIXELS` pixels of the coarser objective.
- `settle_ms`, `save_frames`: as `plate-walls`' (frames under
  `objective-offsets/<token>/<n>/<well>_<side>.tif`).

**Before anything turns or moves**, in this order, each failure a
`PluginRunError` of `objective-offsets`:

1. A dry run: `a dry run cannot turn the turret or image the walls; run it
   without --dry-run`.
2. `confirm_turns` false: `objective-offsets turns the turret to <t1>, <t2>
   with the plate on the stage: check that each objective clears the plate
   (its working distance, the plate's height), then give confirm_turns=true`.
   The tool asks for it on every stand: the profile's
   `turret_requires_confirm` decides what the layer refuses, not what this
   tool asks before it turns several times over a plate.
3. **The plate** (`ctx.upstream("plate", PlateResult)`): its `method` is not
   `wall-images` or `walls` is `None`: `the plate was registered with
   <method>: objective-offsets compares the centres plate-walls measured,
   so register it with plate-walls`; `objective` is `None`: `the plate does
   not record the objective it was registered with: register it with
   plate-walls on a stand with a turret`; `walls.objective` is not
   `objective`: `the plate's record names <a> and its wall images <b>:
   register it again with plate-walls`.
4. **The targets**, against `turret.labels()` without the empty ones: one
   not on the turret, `objective <t> is not on the turret: <labels>`; the
   plate's own, `<t> is the objective the plate was registered with: its
   offset from itself is zero; leave it out of targets`; one given twice,
   `<t> is given twice in targets`.
5. **The wells**: given, each `normalise_well`'d and among the walls' wells
   (`<w> is not a well the registration measured: choose among <A1, H12,
   A12>`), none twice (`<w> is given twice in wells`), at least two (`give
   at least two wells: the offset is a translation only if they agree`);
   not given, every well of `plate.walls.wells` in its order. Their
   `origin_xy_um` are those wells' `center_xy_um`.
6. **The calibrations**: each path through `calibration_from` (§20.3, its
   refusals as they are). Then: one whose `objective` is `None` or not a
   target, `the calibration in <path> is of <objective, or an unknown
   objective>, which is not in targets: give one calibration per target`;
   two of one target, `<path a> and <path b> are both calibrations of <t>:
   give one per target`; a target with none, `no camera calibration of <t>
   in calibrations: turn to it (smc objective set "<t>" --confirm), run smc
   run camera-calibration --run-doc <a document of its own> on a well's wall,
   and add that document to calibrations`. Then, per target,
   `check_calibration(_NAME, calibration, t, True, camera.image_shape())`
   (the frame and the matrix; the objective matches by construction) and
   `check_frame(_NAME, ...)` with the plate's well diameter.
7. **The reference**: `reference = turret.offset(plate.objective).reference`.
   When it is not `None` and not the plate's objective nor a target: `the
   profile's offsets are measured from <R> ([objectives] reference), which
   this run does not measure: add <R> to targets, or clear the reference`.
8. **The walks**, per target: `r` the nominal radius, `search =
   checked_search(_NAME, params.search_um, r, default_um=min(MAX_XY_OFFSET_UM,
   r))`, `step = STEP_FRACTION × min(shape) × calibration.pixel_size_um`,
   `check_walk_size(_NAME, len(wells), search, step)` (per target; with at
   most `MAX_TARGETS` targets the run's frames are bounded in all, FM-140),
   `check_soft_limits(_NAME, xy, <the wells at origin_xy_um>, offsets)`; its
   line, when the stage reports no limits, goes into `assumed` once.

Then `xy.wait()` (FM-85), the start position is read, the clock starts, and
`token = new_token()`.

**Each target, in the order given** (`n` from 1):

1. `ctx.raise_if_stopped()`. **The turn**: `turret.select(t, confirm=True)`.
   A `SafetyRefusedError` is refused as `the turn to <t> was refused: <its
   text>` (raised outside the handler, FM-98); a halt passes unchanged
   (FM-64).
2. **The readback** (FM-167, FM-188): `turret.current()` not `t`: `the
   turret reads <x> after the turn to <t>: it did not turn, or its label
   lags (m2 §8.11 D3); nothing was measured with <t>`.
3. **Z still** (FM-195), when there is a focus drive: `z1 =
   position_um()`, `ctx.wait_s(Z_STILL_S)`, `z2 = position_um()`; when
   `abs(z2 - z1) > Z_STILL_UM` it is refused, `Z moved from <z1> to <z2> µm
   within <Z_STILL_S> s after the turn to <t>: the stand was still moving it;
   nothing was measured with <t>` (`:.2f`, the window `:g`). `z_after_turn_um = z2`. A Z
   read that raises is not caught. `Z_STILL_S = 0.1` and `Z_STILL_UM = 0.5`
   are m2 §8.11 D2's `TURN_Z_SETTLE_S` and `TURN_Z_STILL_UM`, copied with
   that reference: a plugin does not import a backend.
4. **The wells**: `WallWalk(stage="objective-offsets", ...)` from the
   target's calibration and walk, `folder` `<run_dir>/objective-offsets/
   <token>/<n>` with `save_frames`, `no_wall_hint` `check that the light is
   on, that <t> images the wall as the registration's objective did, and
   that search_um (<s> µm) covers the offset`. For each well,
   `measure_well(..., WellRef(well, *origin_xy_um), ...)`: `plate-walls`'
   walk, centred frames, circle fit and refusals (a well of another size, a
   wall that is not a circle) apply under the target too. `on_side` counts
   the sides over the whole run and calls `ctx.progress(i, len(targets) × 4 ×
   len(wells), "<t>: <well> <side>")`.
5. `shift = mean_shift_um(deltas)`, `spread = spread_um(deltas)`, the bound
   `params.max_spread_um` or `default_max_spread_um(plate.walls.pixel_size_um,
   calibration.pixel_size_um)`.

**After the last target**, the stage where the last frame left it:

- Any target whose `spread_um` exceeds its bound: one refusal for all of
  them, so that the numbers are reported and nothing can be written: `the
  wells do not agree on the offset, so it is not a translation or not
  measured well enough: <t>: <w1> (<+dx>, <+dy>), <w2> (...) µm, spread <s>
  µm, bound <b> µm; <the next target>...; look at the frames in <folder>, or
  give a larger max_spread_um if the stage's repeatability is larger`
  (`:+.2f`, the spread and bound `:.2f`; `the frames were not saved` for a
  `folder` that is `None`).
- `offsets_from_reference(plate.objective, shifts, reference)`; an offset
  beyond `±MAX_XY_OFFSET_UM` on either axis: `the offset of <O> would be
  (<x>, <y>) µm, beyond the profile's ±1000 µm: a wrong well or a wrong
  calibration, not two parcentric objectives`.
- `ctx.raise_if_stopped()`, then one move to the last target's last well's
  fitted centre: the operator sees on Live a well centred under the objective
  in place.

**On any exception during the turns and the moves** (a stop, a halt, a fault,
a refusal) nothing is turned back or moved back (FM-88): `warn_left` names
where the stage is, and a second WARNING the turret, `objective-offsets:
the turret is left at <current()>, not turned back` (`current()` in its own
`try`, `unknown` when it fails too); the exception propagates.

**The result**, built once: §20.4's fields, `objective_in_place` the last
target, `elapsed_s` from the first turn to the result. `assumed`, in this
order: the soft limits line (step 8); `the offsets are for the magnifier lever
as it was during this run: the profile keys them by objective alone, for the
lever at 1.0x (FM-103)`; `Z after a turn is where the stand left it once it
was still, not a focused Z: no z_offset_um is measured`; when the profile
names no reference, `the profile names no reference objective: the offsets
are from <plate objective>, which smc objective offset --write makes the
reference`; the saving line of `plate-walls` (`<n> frames could not be saved:
<the first error>`).

**Autofocus, light, Z**: none is set. A turn switches the autofocus off and
the layer never engages it again (m2 §8.3); the stand moves Z on a turn by
itself (m2 §13.1) and the tool reads it only.

### 20.7 Writing the profile — `smc/hardware/profile_write.py` (new), `smc objective offset`

**The block.** The tool writes the profile's `[objectives]` as one block
between two marker lines, and never edits a line outside it (FM-107):

```python
BLOCK_BEGIN = "# >>> [objectives], written by smc objective offset --write (m2 §13)"
BLOCK_END = "# <<< [objectives], written by smc objective offset --write"


def objectives_block(section: ObjectivesSection, *, newline: str = "\n") -> str: ...
def with_objectives(text: str, section: ObjectivesSection, *, where: str) -> str: ...
```

- `objectives_block`: `BLOCK_BEGIN`, the comment line `# The next write
  replaces everything up to the closing marker; edit values here only if no
  write follows.`, `tomli_w.dumps({"objectives": ...})` of `reference` and
  of each entry with its `None` values left out, then `BLOCK_END`; every line
  ends with `newline`.
- `with_objectives(text, section, where=<the file>)`, `ProfileError` naming
  `where` for each refusal: the lines that equal a marker once their line end
  is removed are found. None of either: the block is appended at the end,
  after one empty line (a final line end added first if the text has none).
  One `BLOCK_BEGIN` followed by one `BLOCK_END`: the lines from the first to
  the second, both included, are replaced. Anything else: `the [objectives]
  block that smc writes is damaged (<b> opening and <e> closing marker
  lines): restore it from git, or delete both markers and what lies between,
  then write again`. The text without the block is read by `tomllib`; an
  `objectives` key in it is refused: `the profile has an [objectives] table
  outside the block smc writes: delete it (its values are printed above) or
  edit it by hand instead of writing`. The new text is read back (`tomllib`,
  `Profile.model_validate`) and its `objectives` must equal `section`, else
  `the text written for [objectives] does not read back as the offsets;
  nothing was written` (FM-156). `newline` is `"\r\n"` when the text holds
  one, else `"\n"`: a profile saved by Notepad keeps its line ends.
- `profile.py` gains `profile_text(path: Path) -> str`, the public name of
  `_read_text` (bytes, `utf-8-sig`, FM-80): a profile with a byte-order mark
  is written back without it, which the loader reads the same.

**The command** (m2 §13.5 extended):

```text
smc objective offset [--profile NAME] [--from-run-doc PATH [--session TEXT] [--write]]
```

Without `--from-run-doc` it is m2 §13.5's read-out, unchanged. `--session` or
`--write` without `--from-run-doc` is a usage error (exit 2), as is a
`--session` with a character below `U+0020` (one line in the profile and in
the read-out). With `--from-run-doc` no stand is opened, and in this order,
each failure a `✗` line and exit 1:

1. `Profile.load(profile)`; a profile with no `source` (built in code):
   `<name> is not loaded from a file: give the path of its .toml to -p`.
2. `RunDocument.load(path)`; `unusable("objective-offsets",
   profile=<the profile's name>, dry_run=False)` refused with its text; the
   result validated by `ObjectiveOffsetsResult` (`the objective-offsets
   result in <path> is not one this smc reads; run objective-offsets
   again`).
3. **The reference now**: the profile's `reference` (blank is `None`) must be
   the result's when `reference_in_profile`, and blank when it is not; else
   `the profile's offsets are now from <a>, and the run measured them from
   <b>: run objective-offsets again`.
4. **The section**: the profile's `offsets`, each label of the result's
   `offsets` **replaced** by a new entry (FM-99: not merged; a `z_offset_um`
   it held is dropped and said so) with `xy_offset_um` each value
   `round(v, 2) + 0.0` and `measured = "<finished date, UTC>, <session, >
   objective-offsets in <the run document's name>"` (only the name: FM-50);
   `reference` the result's. `ObjectivesSection.model_validate` of it; a
   `ValidationError` is printed with its text.
5. **The lines**, through `_say` (plain: FM-36), in this order: per target,
   `'<t>'  shift from '<plate objective>' (<+x>, <+y>) µm, spread <s> µm over
   <A1, H12, A12> (bound <b> µm)`; per written label, `'<label>'  xy (<+x>,
   <+y>) µm  was unknown` or `was xy (<+x>, <+y>) µm, measured <m>`; per
   dropped Z, `'<label>': its z_offset_um <+z> µm (measured <m>) is dropped:
   objective-offsets measures XY only`; `reference '<R>'`, with `  (new: the
   profile named none)` when it was blank; `measured = "<text>"`.
6. Without `--write`: `nothing was written: add --write to write these into
   <source>`, exit 0. With `--write`: `profile_text(source)`,
   `with_objectives(...)`, `write_atomically(source, text.encode("utf-8"),
   make_parents=False)` (FM-38), then `✓ wrote [objectives] into <source>;
   smc objective offset -p <profile> reads them back against the turret`,
   exit 0.

The preview is the confirmation the issue asks for: the CLI has no prompt
anywhere (a `--force` or `--overwrite` decides), and a page or a script
cannot answer one. The old values are printed before they are replaced
(FM-172), and the file is in git.

### 20.8 The demo-sample stand's objectives — `[sample] objective_shift_um`

So that a declared offset can be recovered on the demo (the issue's first
criterion) and the page can later be tried on one:

```python
class SampleSection(_Section):  # gains
    objective_shift_um: dict[str, tuple[float, float]] = Field(default_factory=dict)
```

```python
class StandSampleCamera:  # __init__ gains the keyword
    objective_shift_um: Mapping[str, tuple[float, float]] | None = None
```

- **Its meaning**: under the objective of that turret label the camera images
  the sample as if the stage stood at `position − shift`, so a feature is
  centred at its position plus the shift: the shifts are the offsets of m2
  §13.2 from an objective whose shift is zero. A label with no entry has none.
- **Validation** (`ProfileError` naming the key): labels not blank; two
  numbers, finite, not booleans (`_no_boolean`), each within
  `±MAX_XY_OFFSET_UM`. A label that is not on the turret is not checked (the
  profile cannot know it) and has no effect.
- `StandSampleCamera.snap` reads the objective in place (`_objective_name`)
  only when the mapping is not empty, and renders at the shifted position; the
  default path reads no turret and renders as today. `install_sample` passes
  the section's mapping (a copy).
- `sample_lines` gains a fourth line when the mapping is not empty: `  the
  objectives are offset: '<label>' (<+x>, <+y>) µm, ...` (ASCII apart from
  `µ`, FM-20). `sample_refs` stays the sample's own frame (a shift of zero).
- `profiles/demo-sample.toml` gains the key, commented: `# objective_shift_um
  = {}  # turret label -> [x, y] µm: under it a feature is centred at its
  position plus this (m3 §20.8)`. `Profile.demo()` and `demo.toml` are
  unchanged.

### 20.9 Tests

- `tests/unit/test_objective_offsets_algorithm.py` (pure):
  `test_the_shift_is_the_mean_of_the_wells_differences`,
  `test_the_spread_is_the_largest_distance_between_two_wells`,
  `test_one_well_has_no_spread`, `test_no_well_is_refused`,
  `test_the_default_bound_is_two_pixels_of_the_coarser_objective`,
  `test_offsets_are_the_shifts_when_the_profile_names_no_reference`,
  `test_offsets_are_moved_to_a_reference_among_the_targets` (a reference
  whose own shift is not zero, the values written out by hand: FM-203,
  FM-124), `test_a_reference_the_run_did_not_measure_is_refused`.
- `tests/unit/test_profile_write.py`: on the text of
  `profiles/nikon-ti2.toml` read as bytes:
  `test_the_block_is_appended_and_every_other_line_comes_back`,
  `test_a_second_write_replaces_the_block_in_place`,
  `test_crlf_line_ends_are_kept`, `test_a_profile_with_a_byte_order_mark_is_written_without_it`,
  `test_an_objectives_table_outside_the_block_is_refused`,
  `test_a_damaged_block_is_refused` (one marker; two openings),
  `test_a_label_with_quotes_reads_back` (FM-190),
  `test_the_written_text_reads_back_as_the_section`.
- `tests/unit/test_profile.py` gains
  `test_sample_objective_shift_refuses_a_boolean_nan_three_numbers_and_the_bound`
  and the blank label; `tests/unit/test_sample_stand.py` gains the fourth
  line of `sample_lines`.
- `tests/test_demo_sample_demo.py` gains
  `test_an_objective_shift_moves_where_the_sample_is_rendered`
  (`PlateSample.render` patched at class level to record its first two
  arguments, §20.1; frames are not compared, since two snaps at one place
  differ by their noise): under 40X with a shift `(−32.5, 9.75)` the sample
  is rendered at the stage's readback minus the shift, under 10X, with no
  entry, at the readback itself.
- `tests/test_objective_offsets_demo.py` (new), on the demo-sample stand
  (`profiles/demo-sample.toml` with `[sample]` turned as §20.1: camera 30°
  and mirrored, plate 1.2°, shifts 10X `(5.0, −2.5)`, 20X `(0.0, 0.0)`, 40X
  `(−32.5, 9.75)`; the stage's `Velocity` at 100 000): a module fixture runs
  `camera-calibration` under 20X and under 40X into a document each, and
  `camera-calibration` and `plate-walls` (A1, H12, A12, recorded 300 µm off)
  under 10X into `run.json`, through `run_plugin` (the seam, FM-124); each
  test copies `run.json` and opens its own stand.
  `test_the_samples_offsets_are_recovered_within_a_pixel` (targets 40X then
  20X, no reference: `(−37.5, 12.25)` and `(−5.0, 2.5)` within 1.0 µm, one
  pixel of the 10X; written from the shifts, not from the code);
  `test_offsets_are_given_from_the_profiles_reference` (reference 20X:
  10X `(5.0, −2.5)`, 40X `(−32.5, 9.75)`);
  `test_a_reference_the_run_does_not_measure_is_refused_before_the_turret_turns`;
  `test_without_confirm_turns_nothing_turns`;
  `test_a_target_without_a_calibration_is_refused_before_the_turret_turns`;
  `test_a_calibration_of_another_objective_is_refused`;
  `test_a_plate_registered_by_eye_is_refused` (`plate-registration` with
  `sample_refs`); `test_a_well_the_registration_did_not_measure_is_refused`;
  `test_wells_that_disagree_are_refused_with_their_numbers` (H12's
  `center_xy_um` in the plate record moved by `(+20.0, 0.0)` with
  `write_stage`, §20.1: a spread near 20 µm against a bound of 2 µm; the
  record `failed`, no result);
  `test_a_turret_that_reads_another_label_after_the_turn_is_refused` and
  `test_z_still_moving_after_the_turn_is_refused` (the doubles of §20.1;
  nothing imaged under the target); `test_a_stop_leaves_the_turret_and_the_stage_where_they_are`
  (the two WARNINGs); `test_the_stage_ends_at_the_last_wells_centre_under_the_last_target`;
  `test_a_dry_run_is_refused`; `test_frames_are_saved_per_target_without_a_label_in_the_path`.
- `tests/unit/test_slots.py` gains the round trip of the four models.
- `tests/test_cli_objective_offsets.py` (new; a profile file in `tmp_path`
  that is `demo-sample.toml` with the keys above and A1 at the stage origin,
  `a1_center_xy_um = [0.0, 0.0]`; the documents built as in the demo module,
  with A1 and B2 registered: each `smc run` opens a stand at the origin with
  the demo's 10 mm/s, so the wells must be near it; the develop session times
  the module): `smc run objective-offsets` prints the result with
  `offsets`, `spread_um` and the wells; `smc objective offset --from-run-doc`
  prints the lines of §20.7 and leaves the file's bytes as they were;
  `--write` writes them and `smc objective offset` prints each target with
  its `measured` line (the numbers compared within 1.0 µm, not as text:
  `−37.494` is written `-37.49`); a second `--write` replaces the block;
  `--write` without `--from-run-doc` exits 2; a run document of another
  profile, and a reference changed since the run, exit 1 with nothing
  written; `[objectives]` survives in the output (FM-36).
- `tests/test_plate_walls_demo.py` and `tests/unit/test_plate_walls.py`:
  unchanged, and green (§20.2); `tests/test_camera_calibration_demo.py`'s
  reuse tests unchanged (§20.3).

**Failure modes**: FM-14, FM-24, FM-32, FM-36, FM-38, FM-43, FM-50, FM-64,
FM-73, FM-74, FM-80, FM-85, FM-88, FM-98, FM-99, FM-103, FM-107, FM-121,
FM-124, FM-125, FM-140, FM-142, FM-156, FM-167, FM-172, FM-188, FM-189,
FM-190, FM-195, FM-203.

### 20.10 Not verified (for the owner)

- **At the Ti2-E**: the offsets 4× → 10× and 4× → 40× measured on a plate,
  and compared once with m2 §13.6's procedure by hand; whether the 40×
  images a well's wall in brightfield as the 4× and 10× do (its depth of
  field, the meniscus), so that `find_wall` finds it; whether 2 pixels of the
  4× (5.5 µm) hold the wells' spread, which also carries the stage's
  repeatability; how long a run takes with the turns and the moves between
  wells.
- That the shift is a translation over the plate (m2 §13.8): two wells far
  apart that agree within the bound are the evidence the tool gives.
- Whether a turn at the plate's working height (Z about 4195 µm at 10×)
  settles within the turret's wait (m2 §8.11.3); the tool refuses a Z that
  still moves and says so.

## 21. The focus on the cells (#341)

PFS locks on the plate's bottom and holds the plane its offset names. The
cells sit some micrometres above that plane, and the sharpest plane differs
from well to well and from plate to plate. This section adds a software
autofocus: a sweep of a few planes in Z around a centre, a focus score on
each frame, and the plane of the best score. It is a tool of its own,
`focus-on-cells`, and an optional step of `scan` (§8). **Merging this
section decides it.** It extends §8 additively (new parameters, one new
optional field of `ScanFrame`) and changes no layer interface: the sweep
uses `ZStage`, `Camera` and `Autofocus` as they are (m2 §3, §12).

### 21.1 What was measured (2026-11-09, planning #341)

On `main` `188c6d1`, Python 3.13, pymmcore-plus 0.18.1, numpy 2.5.3, in
throwaway scripts on `profiles/demo-sample.toml` (not committed):
- **The demo-sample's defocus.** `PlateSample.render` blurs by a Gaussian
  of `|z| / 10` pixels (capped at 32), sharp at Z = 0, symmetric. The noise
  is drawn per snap. The texture covers the plastic too, so no field of
  the demo is flat.
- **Four scores over Z = −30…30 µm, step 2, in A1** (definitions in §21.3):
  each peaks at Z = 0. Brenner rises 2.9× from the ends to the peak,
  normalised variance 3.3×, gradient energy 2.0×, and the variance of the
  Laplacian only 1.49×: at the demo's noise (σ = 20 counts) the Laplacian
  measures mostly noise.
- **The parabolic refinement** (§21.3) on three repeats of each sweep, in a
  well and on the plastic: ±5 µm step 1 centred on 0 or on 1.3, ±25 µm
  step 5 centred on 20, ±30 µm step 2 centred on 0. Every estimate lay
  within 0.30 µm of 0, for each of Brenner, normalised variance and
  gradient energy. The sweep of the issue's criterion, from Z = 20 µm with
  `range_um = 25`, `step_um = 5`, gave −0.01 to 0.03 µm.
- **The relative rise** `(max − min) / max` of a sweep: 0.17–0.31 for ±5 µm
  step 1, 0.49–0.75 for the wide sweeps. No flat sweep exists on the demo
  (see above); §21.3's flat bound is chosen.
- **Time on the demo**: a Z move returns in under 1 ms, a snap of 512 ×
  512 takes about 20 ms, so 31 planes took 0.8 s. A score of one frame is a
  few milliseconds. Nothing was timed at the Ti2-E.
- **The demo's autofocus** (`DAutoFocus`): off when the stand opens;
  `engage()` returns `True` and moves no Z; `ZStage.move_to_um(z,
  keep_autofocus=False)` switches it off and leaves it off; `engage()`
  again returns `True` with Z where it was.
- **The scan's own autofocus double** (`tests/test_scan_demo.py`'s
  `_Autofocus`) is installed with `Microscope.override`, so the Z interlock
  does not reach it (m2 §3.4: an overridden autofocus is not interlocked):
  the sweep must switch it off itself (§21.4).
- **Nothing in `src/` or the sibling repositories scores focus**: there is
  no code to port. Session 3's 40× stacks, which the issue names, are not
  taken yet (`local/surveys/nikon-ti2/` holds sessions 0 to 2).

### 21.2 Decisions

**D1. The sweep moves Z with the autofocus off, and the frame is taken at
the best plane with the autofocus off.** The 40×'s `offset_per_um` factor
is not measured (m2 §12.10.9), so `focus_by_um` refuses under lock (m2
§3.5), and a step through the offset under lock costs a relock and its
settle window (1 s on the Ti2-E) per plane. With PFS off, a Z move is a
plain move under the soft limits, and an engage at the offset the sweep
started from is an engage at a seen offset, inside D2's band (m2 §12.2):
the sweep's Z stays within `range_um` of the plane PFS held. So, when the
autofocus is used: read the lock, switch it off, sweep, move to the best
plane, snap; the next field's existing re-engage (§8) locks it again, and
PFS takes Z back to its plane, which carries the plate's tilt to the next
field. The frame's `z_um` is then the cells' plane, and its `pfs_offset`
the offset of the lock the sweep was centred on.
- Not chosen: *the offset under lock* (`focus_by_um`). It needs the factor,
  and costs a settle window per plane. Once the 40×'s factor is in the
  profile, carrying a well's `dz_um` into the offset once (so that PFS holds
  the cells' plane for the rest of the well, with no cost per field) is a
  later change, measured against this one's cost per field.
- Not chosen: *a sweep with PFS on* through Z moves: the Z interlock
  switches PFS off for each move and relocks, which takes Z back to PFS's
  plane before the snap.

**D2. A sweep is symmetric around its centre, in planes `step_um` apart,
in increasing Z.** The centre is the Z PFS holds at the field when the
autofocus is used, else the Z read before anything moved (the operator's
focus). It never follows the previous field's best plane, so nothing
drifts from field to field. The direction of "toward the sample" plays no
part: at 40× the working distance is 170–250 µm (m2 §12.10.1), and
`range_um` is at most 50 µm and at most the stand's Z jog limit.

**D3. Bounds before the first move.** A sweep runs only when every plane
lies inside `ZStage.limits_um()`; a stand with no Z soft limits refuses
the focus on the cells (unknown is not unlimited). `range_um` must not
exceed the stand's Z jog limit (`profile.safety.max_z_jog_um`, else the
layer's `DEFAULT_MAX_Z_JOG_UM`): the sweep's absolute moves are not
jog-guarded by the layer (M1 §7), so the plugin bounds their reach with
the same number. Every Z readback is compared with its command (FM-167).

**D4. A peak, or the centre is kept.** The score's best plane is taken
only when the sweep has one clear interior maximum. A flat sweep (an empty
field), a best at an end of the sweep, or two maxima (brightfield of
unstained cells can lose contrast exactly at focus and show a maximum on
either side) keep the centre, and say so. Unknown beats guessed: which of
these the real 40× stacks show is session 3's (§21.10).

**D5. Per field or per well.** `per-field` sweeps every field.
`per-well` sweeps the well's fields in order until one finds a peak, then
moves every later field of the well to its own centre plus that field's
`dz_um`, with no sweep. An empty first field (flat) therefore does not
decide a well.

**D6. Offline first, with the tool's own stacks.** `focus-on-cells` keeps
every frame of its sweep as a TIFF with its Z, so a run at a stand is a
Z-stack. `scripts/dev/focus_scores.py` scores such stacks with every
score of §21.3 and prints each curve and peak, so the default score and
the sweep's defaults are chosen on real 40× stacks before the scan uses
the step. Until then the defaults are chosen, not measured, and labelled.
The tool's own run at the Ti2-E is the stack: the sweep stays within
`range_um` of a plane PFS held, with the relock of D1.

### 21.3 Algorithm — `smc/plugins/cell_focus/algorithm.py`

Pure: numpy only, no capability.

```python
ScoreName = Literal[
    "brenner", "normalized-variance", "gradient-energy", "laplacian-variance"
]
PeakOutcome = Literal["peak", "flat", "at-edge", "two-peaks"]

#: Chosen from the literature, not measured on cells: session 3's stacks
#: decide (m3 §21.10). On the demo-sample all four peak at Z = 0.
DEFAULT_SCORE: ScoreName = "brenner"
#: The widest half-range of a sweep. Chosen: the 40×'s working distance is
#: 170-250 µm (catalogue), and a sweep never needs more than a few µm.
MAX_RANGE_UM = 50.0
#: Chosen: below the Ti2-E's Z steps that were read (0.01 µm) times ten.
MIN_STEP_UM = 0.1
MAX_STEP_UM = 10.0
#: Bounds the frames of one sweep (FM-74). Chosen.
MAX_PLANES = 101
#: A sweep whose (max - min) / max is below this is flat. Chosen: the demo
#: gives 0.17 at its narrowest sweep; no real flat sweep was read.
MIN_RISE = 0.05
#: A second interior maximum whose rise above the sweep's minimum reaches
#: this fraction of the best's makes the sweep "two-peaks". Chosen.
SECOND_PEAK_FRACTION = 0.5
#: A Z readback farther from its command than this fraction of the step
#: fails the sweep (FM-167). Chosen.
LANDING_FRACTION = 0.25


@dataclass(frozen=True)
class Peak:
    outcome: PeakOutcome
    best_index: int
    z_um: float | None  # the refined plane; None unless outcome == "peak"


def score_frame(frame: npt.NDArray[Any], name: ScoreName) -> float: ...
def plan_planes(
    centre_um: float, range_um: float, step_um: float
) -> tuple[float, ...]: ...
def first_outside(
    planes_um: Sequence[float], low_um: float, high_um: float
) -> float | None: ...
def find_peak(z_um: Sequence[float], scores: Sequence[float]) -> Peak: ...
```

**Scores.** The frame is read as `float64`; `m` is its mean. A frame that
is not 2-D or smaller than 3 × 3 is a `ValueError`. A mean that is not
finite and above 0 (a dark frame) scores `0.0`. Each score is
dimensionless or in counts, and only compared within one sweep:
- `brenner`: `mean((f[:, 2:] - f[:, :-2]) ** 2) / m**2`;
- `gradient-energy`: with `gx = f[:, 1:] - f[:, :-1]` and
  `gy = f[1:, :] - f[:-1, :]`,
  `mean(gx[:-1, :] ** 2 + gy[:, :-1] ** 2) / m**2`;
- `laplacian-variance`: the variance of `4 f[1:-1, 1:-1]` minus its four
  neighbours, `/ m**2`;
- `normalized-variance`: `var(f) / m`.

**`plan_planes`**: `k = floor(range_um / step_um + 1e-9)`; the planes are
`centre_um + i * step_um` for `i` in `-k…k`, in increasing order. A
`range_um` or `step_um` that is not finite, `k < 1`, or `2k + 1 >
MAX_PLANES` is a `ValueError` naming the numbers (the parameters' own
validators refuse them first, §21.5).

**`first_outside`** returns the first plane outside `[low_um, high_um]`,
or `None`.

**`find_peak`.** `z_um` strictly increasing, the same length as `scores`,
at least 3, every value finite, else `ValueError`. With `i` the first index
of the maximum, in this order:
1. `(max - min) / max < MIN_RISE`, or `max <= 0`: `flat`;
2. `i` is 0 or the last: `at-edge`;
3. another interior index `j`, with `scores[j]` strictly above both its
   neighbours and `scores[j] - min >= SECOND_PEAK_FRACTION * (max - min)`:
   `two-peaks`;
4. otherwise `peak`, and `z_um` is the vertex of the parabola through the
   points `i - 1`, `i`, `i + 1` (unequal spacing allowed: the readbacks),
   clamped to `[z[i - 1], z[i + 1]]`; when the three are collinear or
   convex, `z[i]`.

`best_index` is `i` in every outcome.

### 21.4 The sweep — `smc/plugins/cell_focus/loop.py`

Shared by the tool and the scan.

```python
class FocusDriveOffTarget(Exception):
    """A Z readback farther from its command than the sweep allows."""
```

```python
class AutofocusStillOn(Exception):
    """The autofocus still reads on after the sweep switched it off."""
```

```python
def switch_off_for_sweep(af: Autofocus) -> None: ...
def sweep(
    z: ZStage,
    camera: Camera,
    planes_um: Sequence[float],
    step_um: float,
    score: ScoreName,
    *,
    raise_if_stopped: Callable[[], None],
    on_plane: Callable[[int, float, npt.NDArray[Any]], None] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]: ...
def move_checked(z: ZStage, target_um: float, tolerance_um: float) -> float: ...
```

Neither exception is a `HardwareError`: each names what the sweep saw,
and its callers catch both by name.

- **`switch_off_for_sweep`**: `af.disengage()`, then `af.is_engaged()`
  must read `False`, else `AutofocusStillOn`: `the autofocus still reads on
  after it was switched off for the focus sweep: nothing was moved`. The
  explicit switch-off is needed because an overridden autofocus is not
  interlocked (§21.1), and it is what D1 asks on every stand. Never called
  in a dry run: `disengage()` is a safe call that acts in dry run too (m2
  §3.2).
- **`sweep`**: for each plane, `raise_if_stopped()`, then
  `move_checked(z, plane, LANDING_FRACTION * step_um)`, then
  `camera.snap()`, the score, and `on_plane(k,
  readback, frame)` (the tool writes the frame there, so a failure later
  keeps the frames taken). It returns the readbacks and the scores.
  Every move is `move_to_um(plane, keep_autofocus=False)`.
- **`move_checked`**: `move_to_um(target, keep_autofocus=False)`, and a
  readback with `not abs(readback - target) <= tolerance` (a NaN fails,
  FM-104) raises `FocusDriveOffTarget`: `Z read <r> µm after a move to <t>
  µm, more than <tol> µm away` (two decimals).
- **Errors pass through**: the sweep catches nothing. A
  `MicroscopeHaltedError`, a `MotionStoppedError` and `StopRequested` reach
  the caller as they are (FM-64); so does any `HardwareError`. Nothing is
  restored after a failure (FM-88): Z stays where it was read last, and
  the autofocus stays off.

### 21.5 The slot — `smc/plugins/slots/cell_focus.py`

```python
class FocusSweep(SlotValue):
    score: ScoreName
    centre_um: float
    z_um: tuple[float, ...]  # the readbacks, increasing
    scores: tuple[float, ...]
    outcome: PeakOutcome
    best_z_um: float | None  # set when outcome == "peak", else None
```

```python
class FieldFocus(SlotValue):
    source: Literal["swept", "well"]
    sweep: FocusSweep | None  # set when source == "swept", else None
    dz_um: float  # the frame's Z command minus the centre; 0.0 without a peak
    frame_score: float  # the score of the frame written, by sweep's score
    duration_s: float = Field(ge=0.0)
```

```python
class CellFocusResult(PluginResult):
    folder: str  # "cell-focus/<token>", relative, POSIX
    objective: str | None
    x_um: float | None  # the XY readback, None without an XY stage
    y_um: float | None
    autofocus: Literal["locked", "off", "none"]  # at the start
    pfs_offset: float | None  # the lock's offset, when it was used
    sweep: FocusSweep
    files: tuple[str, ...]  # one per plane, in the sweep's order
    best_file: str | None  # the frame at the best plane, None without one
    z_um: float  # Z read at the end
    relocked: bool | None  # None when the autofocus was not used
```

Validators (each a `ValueError` naming the fields): `z_um` and `scores`
the same length, at least 3; `best_z_um` set exactly when `outcome ==
"peak"`; `sweep` set exactly when `source == "swept"`; `files` empty (a
dry run) or as long as `sweep.z_um`. All four are exported from `slots/__init__.py`.

`ScanFrame` gains `cell_focus: FieldFocus | None = None`, last. **A frame
without it is written exactly as before**: a wrap serializer drops the key
when it is `None`, as `StageRecord.preset` does (§1.7), so a scan without
the step still reads in an older smc and `SCHEMA_VERSION` stays 1. A scan
with the step does not read in an older smc, which refuses the key
(FM-107), as intended.

### 21.6 In the scan — `scan/plugin.py`

```python
class ScanParams(PluginParams):  # gains, last
    cell_focus: Literal["off", "per-field", "per-well"] = "off"
    focus_range_um: float = Field(5.0, gt=0.0, le=MAX_RANGE_UM)
    focus_step_um: float = Field(1.0, ge=MIN_STEP_UM, le=MAX_STEP_UM)
    focus_score: ScoreName = DEFAULT_SCORE
```

A model validator refuses `focus_step_um > focus_range_um` and more than
`MAX_PLANES` planes, naming both. The defaults are chosen (11 planes,
±5 µm): session 3 measures them (§21.10). A `--resume` of a scan stopped
by an smc without these keys is refused as "the parameters differ"
(the runner's rule, §1.8): run it again without `--resume`.

**Before anything moves** (after §8's checks, dry run included), when
`cell_focus != "off"`:
1. a `ZStage`, else `PluginRunError`: `cell_focus needs a focus drive, and
   this stand has none`;
2. `z.limits_um()` not `None`, else `PluginRunError`: `cell_focus moves Z:
   the profile must set [safety] z_soft_limits_um`;
3. `focus_range_um` not above the Z jog limit (D3), else `PluginRunError`
   naming both;
4. `z_start = z.position_um()`, and `plan_planes(z_start, …)` inside the
   limits, else `PluginRunError`: `the focus sweep around <z_start> µm would
   reach <plane> µm, outside the soft limits [<lo>, <hi>] µm`.

**A resume** of a scan whose checkpoint holds a frame with `focus ==
"locked"`, when the autofocus is not used now and `autofocus != "off"`,
is a `PluginRunError`: `the scan being resumed used the autofocus, which is
not locked now (a stop during the focus on the cells leaves it off): lock
it on the sample, then run scan --resume`. Without this, `if-engaged`
would resume silently without PFS.

**Per field**, after §8's lock handling and before the Z and offset reads,
when `cell_focus != "off"` and not in a dry run:
1. **No step** when the autofocus is used and the field reads `unlocked`
   (its Z is where a search left it, FM-154): `cell_focus` stays `None`,
   counted for `assumed`.
2. `centre = z.position_um()` when the autofocus is used, else `z_start`;
   `pfs_offset` is read here, before the switch-off (the lock the sweep
   was centred on).
3. `per-well` with the well's `dz_um` known: no sweep; the target is
   `centre + dz_um`. Otherwise the planes are `plan_planes(centre, …)`,
   and a plane outside the limits fails the field with nothing moved for
   it: `SweepOutsideLimits: the focus sweep around <c> µm would reach <p>
   µm, outside the soft limits [<lo>, <hi>] µm`.
4. When the autofocus is used: `off_by_focus = True`, then
   `switch_off_for_sweep(af)`.
5. The sweep (§21.4), then `find_peak`. With a peak, the target is its
   `z_um`, and `per-well` keeps `dz_um = target - centre` for the well's
   later fields; without one, the target is the centre.
6. `move_checked(z, target, LANDING_FRACTION * focus_step_um)`. Then §8's
   `z_um` read, the snap, `frame_score = score_frame(frame, focus_score)`,
   and `FieldFocus(..., duration_s=…)` from `time.perf_counter()` taken
   before step 4 and after the snap.
- `FocusDriveOffTarget`, `AutofocusStillOn` and every `HardwareError` of
  the step are a failed field (`<Type>: <message>`, §8), and a WARNING
  names the field and says the autofocus is left off when step 4 ran.
  `MicroscopeHaltedError`, `MotionStoppedError` and `StopRequested` pass.
- **The next field's lock** is §8's re-engage, whose condition becomes
  `off_by_the_scan or off_by_focus or af.is_engaged()`. `off_by_focus` is
  cleared as `off_by_the_scan` is (a field that reads locked, FM-129). The
  engages that follow a focus step are timed (`time.perf_counter()`) and
  counted apart: §8's `assumed` line keeps counting only the others.
- **After the last field**, when the scan ends without an error and
  `off_by_focus` is set: one `engage()`. `False`, or a `HardwareError`, is a
  WARNING and an `assumed` line (`the autofocus did not lock again after the
  last field (<why>): it is off`), not a failure: every frame is written.
  A failure or a stop leaves it off (FM-88), and §8's `PluginRunError`
  texts gain `; the focus on the cells left the autofocus off` when
  `off_by_focus` is set.

**`assumed`** gains, when the step ran:
- `the focus on the cells added <a> s per field (<n> fields: <b> s to
  sweep and move, <c> s for the lock after)`, means to two decimals; `<c>`
  is left out without the autofocus;
- `the focus on the cells found a peak on <p> of <s> sweeps (flat <f>, at
  an edge <e>, two peaks <t>): the others kept the centre`;
- `the focus on the cells was skipped on <u> fields whose autofocus did not
  lock`, when `u > 0`;
- with the autofocus used: `each frame's pfs_offset names the plane PFS
  held, not the cells' plane <min>…<max> µm from it: an export that
  engages PFS at that offset images the plane PFS held` (route and
  NIS-Elements: #392);
- in a dry run, instead: `dry run: the focus on the cells was planned
  (<n> planes per sweep), not run`.

### 21.7 The tool — `smc/plugins/cell_focus/plugin.py`

Entry point `focus-on-cells = "smc.plugins.cell_focus.plugin:FocusOnCellsPlugin"`.
`requires = (ZStage, Camera)`, `provides = "cell-focus"`, `reads = ()`,
optional `XYStage`, `Autofocus`, `ObjectiveTurret` through `get`.

```python
class FocusOnCellsParams(PluginParams):
    range_um: float = Field(5.0, gt=0.0, le=MAX_RANGE_UM)
    step_um: float = Field(1.0, ge=MIN_STEP_UM, le=MAX_STEP_UM)
    score: ScoreName = DEFAULT_SCORE
    relock: bool = True
```

The same model validator as the scan's. It sweeps where the stage is; it
never moves XY.

**Before anything moves**, in this order, dry run included: the autofocus,
when the stand has one and it reads engaged, must read locked, else
`PluginRunError`: `the autofocus is on and not locked: lock it on the
sample, or switch it off to sweep around the Z where it is`; used means
engaged and locked, and then its offset is read. Then §21.6's checks 2–4
around `z.position_um()`. The token is `new_token()`, the folder
`cell-focus/<token>/`, every path through `resolve_in_run`, and the
Windows `MAX_PATH` check of §8 on the longest one.

**The run**: `switch_off_for_sweep` when used; the sweep, each frame
written by `on_plane` with `write_atomically` as a TIFF in its own dtype,
`<k:03d>.tif` (k from 0), with `{"z_um", "z_command_um", "score",
"score_name"}` as tifffile's JSON metadata (§8's encoding); then
`find_peak`. With a peak: `move_checked` to it, a snap written as
`best.tif` and scored. Without one: `move_checked` back to the centre, no
best frame. Then, when used and `relock`: `engage()`; `relocked` is its
return value, and `False` adds `the autofocus did not lock again: it is
off` to `assumed` with a WARNING. Used and not `relock`: `assumed` gains
`the autofocus was left off, at the best plane` (or `at the centre`).
`z_um` is read last.

**Errors**: a failure during the sweep or the moves (`HardwareError`,
`FocusDriveOffTarget`, `AutofocusStillOn`) is a `PluginRunError`, raised
outside the handler (FM-98): `the focus sweep ended at plane <k> of <n>
(<Type>: <message>); the <m> frames taken are kept in
cell-focus/<token>/[; the autofocus was left off]; Z was left where it
stopped`, with `k` counted from 1 and `m` the frames written. A failure
after the sweep (the move to the best plane, its snap) names that step
instead of a plane.
An `OSError` writing a frame is §8's `AtomicWriteError` text. Halt, stop
and `StopRequested` pass. No checkpoint: a run is one sweep, and
`--resume` starts again.

**Dry run**: the checks, then each plane walked with `move_checked`
through the layer's dry run (whose readback is the command), with no
snap, no `sweep()` call and no switch-off. The result's `FocusSweep` has
the planes as `z_um`, every score `0.0`, `outcome` `"flat"`; `files` is
empty, `best_file` and `relocked` are `None`, and `assumed` gains `dry run:
<n> planes walked, no frame taken`.

### 21.8 The offline script — `scripts/dev/focus_scores.py`

`python scripts/dev/focus_scores.py <folder>... [--csv out.csv] [--plot
dir]`. Each folder is one stack: the TIFFs in it whose JSON metadata has
`z_um` (the tool's frames; `best.tif` is left out). For each stack it
prints one row per plane (Z, the four scores) and, per score, the outcome
and the refined Z of `find_peak`. `--csv` writes every row with its
folder; `--plot` writes one PNG per stack (score normalised to its
maximum, against Z) when `matplotlib` imports, and says so and goes on
when it does not. A folder with fewer than 3 such frames is reported and
skipped. Exit 0 when one stack was scored, else 1. It imports
`smc.plugins.cell_focus.algorithm`, so the script and the tool score
alike. UTF-8 everywhere, `pathlib`, no shell.

### 21.9 Tests

- **Algorithm** (`tests/unit/test_cell_focus_algorithm.py`, synthetic
  stacks from `PlateSample().render` at known Z, `noise_std` default and
  0):
  - `test_each_score_peaks_at_the_sharp_plane` (all four, Z = −20…20 step
    2, the argmax at Z = 0);
  - `test_the_peak_is_refined_between_planes` (a sweep centred on 1.3 µm,
    step 1: the refined Z within 0.3 µm of 0, and not equal to a plane);
  - `test_a_flat_sweep_keeps_no_peak`, `test_a_best_at_the_end_is_at_edge`,
    `test_two_maxima_are_two_peaks`, on score lists written out by hand,
    each from both sides of its bound (`MIN_RISE` at 0.04 and 0.06,
    `SECOND_PEAK_FRACTION` at 0.4 and 0.6, FM-73);
  - `test_unequal_spacing_is_refined_through_the_readbacks`;
  - `test_plan_planes_is_symmetric_and_bounded`, `test_too_many_planes_is_refused`,
    `test_first_outside_names_the_first_plane_past_a_limit`;
  - `test_a_dark_frame_scores_zero`, `test_find_peak_refuses_nan_and_unsorted_z`.
- **The tool** (`tests/test_cell_focus_demo.py`, `profiles/demo-sample.toml`):
  - `test_a_sweep_from_20_um_finds_the_sharp_plane_within_one_step`
    (`range_um=25`, `step_um=5`: `best_z_um` within 5 µm of 0 and Z ends
    there; measured 0.03);
  - `test_a_sweep_past_the_soft_limit_is_refused_before_any_move` (Z at
    990, `range_um=25`: `PluginRunError`, Z still 990, no file);
  - `test_a_guard_trip_ends_the_sweep_and_keeps_the_frames` (a `ZStage`
    override whose third move raises `SafetyRefusedError`: the error names
    plane 3 of 11, two files kept, no `engage()` after);
  - `test_a_locked_autofocus_is_switched_off_for_the_sweep_and_relocked`
    (the demo's `engage()` first: off during every move, locked after,
    `relocked is True`);
  - `test_relock_false_leaves_it_off_at_the_best_plane`;
  - `test_an_autofocus_on_and_not_locked_is_refused`;
  - `test_a_halt_during_the_sweep_is_not_a_plugin_error` (FM-64);
  - `test_a_readback_off_its_command_fails_the_sweep` (FM-167);
  - `test_a_dry_run_moves_nothing_and_writes_nothing`.
- **The scan** (`tests/test_scan_demo.py`, its rig; the `_Autofocus`
  double gains a plane it drives Z to on `engage()`, e.g. 3.0 µm, while the
  `SampleCamera` is sharp at 0):
  - `test_per_field_focus_takes_each_frame_at_the_sharp_plane` (every
    frame's `z_um` within one step of 0, `dz_um` near −3, the autofocus
    off at every snap and locked after the scan);
  - `test_per_well_focus_sweeps_until_a_peak_then_reuses_its_dz` (the
    first field made flat by its camera: the second is swept, the later
    ones `source == "well"`);
  - `test_an_unlocked_field_gets_no_focus_step`;
  - `test_a_failed_sweep_is_a_failed_field_and_leaves_the_autofocus_off`;
  - `test_the_next_field_relocks_after_the_focus` (the engage count, and
    §8's `assumed` line not counting them);
  - `test_the_focus_time_per_field_is_reported` (the line, not a value);
  - `test_cell_focus_without_soft_limits_is_refused_before_anything_moves`;
  - `test_a_resume_after_the_autofocus_was_left_off_is_refused`;
  - `test_a_frame_without_cell_focus_is_written_as_before` (no key in the
    dump; `tests/unit/test_slots.py`'s `SAMPLES` gain a frame with it).
- **Script** (`tests/test_dev_focus_scores.py`): a stack written by the
  tool on the demo-sample, scored: exit 0, a row per plane, the peak near
  0; a folder of two frames is skipped and exit 1.
- The presets' round-trip case for `focus-on-cells`
  (`tests/unit/test_preset_from_document.py`) and the registry's listing.

**Failure modes**: FM-12, FM-19, FM-24, FM-32, FM-38, FM-43, FM-64,
FM-73, FM-74, FM-88, FM-98, FM-104, FM-111, FM-118, FM-124, FM-129, FM-142,
FM-154, FM-155, FM-165, FM-167, FM-182, FM-195.

### 21.10 Not verified (for the owner)

At the Ti2-E, 40× on a plate with cells, PFS locked on them (m2 §12.10.7):
- `smc run focus-on-cells --param range_um=5 --param step_um=0.5` on a
  few wells, each run's folder copied to `local/`; then
  `scripts/dev/focus_scores.py` on them: which score peaks at the by-eye
  focus (within 1 µm), whether the curves show two maxima (D4), and the
  defaults of range and step.
- The PFS lock held after the relock (`smc autofocus get`), and how far Z
  moved in it.
- The time per plane (a Z move of a step and its settling, FM-195, then a
  snap) and per field, against §21.6's `assumed` line.
- Whether a Z readback lands within a quarter step of its command.
- The 40×'s offset factor, which #392 (route and NIS-Elements) waits for.

## 22. The lab's own plate definitions (#400)

Every plate tool names its plate by a `useq` key (`24-well`), and smc took
only the keys `useq` ships. The Greiner 24-well plate on the Ti2-E is not
`useq`'s `24-well`: its pitch is 19.45 mm, not 19.0 (F9, session 3, #405),
so `calibrate` refuses it at +2.4 % (`SCALE_TOLERANCE = 0.02`, §17). This
section adds the lab's own definitions, kept in one versioned data file and
registered with `useq` when `smc` is imported, so that every entry point (the
CLI, the page, the plugins, a run document read back) sees them under their
names. **Merging this section decides it.** It is additive: no slot, model or
parameter changes, and a plate tool reads a lab plate as it reads `96-well`.

### 22.1 What was measured (2026-11-10, planning #400)

On main at `caaa60e` (useq-schema 0.9.2, pydantic 2.13, Python 3.13), with a
prototype of §22.3 in the design session's scratchpad:

- `useq.register_well_plates({name: {rows, columns, well_spacing,
  well_size, circular_wells}})` updates a module-level dict; afterwards
  `useq.WellPlate.from_str(name)` and `useq.registered_well_plate_keys()`
  see the plate (`well_spacing=(19.45, 19.45)`, `well_size=(15.6, 15.6)`,
  `name` the key). A key already present is **overwritten** without a word.
- Session 3's hand centres, B3 (10 003, -10 051), B5 (-28 982, -9 919), C3
  (10 031, 9 375) µm: under `24-well`, refused at +2.4 %; under a plate at
  19.45 mm, the scale passes (+0.07 %) but the fit turns the plate by
  179.8° and `max_rotation_deg` refuses it. The plate lay **turned 180° in
  its holder** (owner, 2026-11-10: row A at the front, by the operator, and
  row D at the back), so its columns ran towards −X and its rows towards +Y.
  Put the right way, they run as `useq`'s (+X, −Y): no orientation of the
  stand is needed. That the refusal blames a misnamed well rather than a
  turned plate is **#414**. The same centres turned 180° about the origin fit
  at −0.21°, **residual 44.5 µm**.
- Synthetic centres of A1, A6 and D1 at 19.45 mm, turned 0.3°: residual 0
  under the new plate, refused at +2.4 % under `24-well`.
- The plate-related suites (plate-finding, -registration, -walls, presets,
  the guide, the demo sample) with the plate registered before collection:
  one test changes. A synthetic survey of `useq`'s `24-well` is identified
  as `greiner-24-well`, 0.809 against 0.804, and `decide` says `barely
  beats` (the decisive margin is 0.08); a survey of the Greiner plate goes
  the other way, 0.803 against 0.799. Both fits place every well within
  1.9 mm, under the test's bound of 30 % of a radius (2.3 mm).
- A wheel built from main with a file at `src/smc/plates.toml` ships it as
  `smc/plates.toml`: hatch takes every file of the package.

### 22.2 Decisions

**D1. One data file, shipped in the package: `src/smc/plates.toml`.**
Plates are the lab's, not a stand's: a profile (`profiles/*.toml`, ADR-0003)
is the wrong home, and a Greiner plate goes from the Ti2-E to another stand
unchanged. Package data travels with the wheel and with an editable install,
and needs no search path. A stand-local file (an `$SMC_PLATES`) is not added:
nothing asks for it yet.

**D2. Every number says where it comes from.** `well_spacing_mm` and
`well_size_mm` each carry a source: `measured` (where, when, how, the issue),
`maker` (the maker's reference and datasheet) or `assumed` (taken from
elsewhere, not checked on this plate). Rows and columns are counted, not
sourced.

**D3. Registered at `import smc`.** `smc/__init__.py` calls
`smc.plates.register_lab_plates()`. Every entry point imports `smc` first,
so no call site needs to remember it, and a run document that names a lab
plate validates in any process that reads it. `useq` is imported there
(~0.1 s, already paid by `smc.hardware`, `smc.cli` and the plugins).

**D4. A lab plate never redefines a `useq` key.** `useq` overwrites a key
silently (§22.1): a lab file that names `24-well` would change the geometry
of every run document that holds it. The registration refuses a name `useq`
already holds, with a `ValueError` that names the file; a second call is a
no-op that returns the same names.

**D5. A name is published once: its numbers do not change under it.** A run
document stores the plate's name, not its geometry, so editing the spacing
of `greiner-24-well` would move every well of every plate registered with it
(0.75 mm at the sixth column for 0.15 mm of pitch), with nothing to say so.
A corrected definition (the maker's datasheet replacing the measurement)
takes a **new name**, and the old one stays in the file, its description
saying what replaced it. A unit test pins each published name's numbers, so
an edit in place fails the suite until its test is changed in the same
commit, deliberately.

**D6. Names are `useq`-style and end in `-well`**: lower case, digits and
hyphens, `[a-z0-9]+(-[a-z0-9]+)*-well`. The profile's `[sample] plate`, the
page's menu and `plate-finding` select well plates by that suffix (§2, m1
§8). The first one is `greiner-24-well`.

**D7. `plate-finding` fits lab plates too.** Its default candidates are
"every registered `-well` plate" (§2), and a lab plate is one. A survey of a
24-well plate then weighs `24-well` against `greiner-24-well`, 2.4 % apart,
and it cannot tell them apart (§22.1): its decisive-margin note says so,
which is the truth, and the operator names the plate. A survey that knows
its plate passes `plates=[...]`. Hiding a lab plate from the default would
instead fit a Greiner plate as `24-well`, 2.25 mm off at the sixth column,
with no note.

### 22.3 `src/smc/plates.toml` (new)

```toml
# The lab's own plate definitions (m3 §22). Read when smc is imported and
# registered with useq under these names. A name, once used in a run
# document, keeps its numbers: a corrected plate takes a new name (D5).

[plates.greiner-24-well]
description = "Greiner Bio-One 24-well plate, standard bottom, catalogue reference not read yet"
rows = 4
columns = 6
well_spacing_mm = [19.45, 19.45]
well_size_mm = [15.6, 15.6]
circular_wells = true
well_spacing_source = { kind = "measured", note = "..." }
well_size_source = { kind = "assumed", note = "..." }
```

The notes are the plan's (#400). Keys outside the model are refused.

### 22.4 `src/smc/plates.py` (new)

```python
PLATES_FILE = "plates.toml"


class PlateSource(BaseModel):  # frozen, extra="forbid"
    kind: Literal["measured", "maker", "assumed"]
    note: str  # min_length=1


class LabPlate(BaseModel):  # frozen, extra="forbid"
    description: str  # min_length=1
    rows: int  # 1..64
    columns: int  # 1..96
    well_spacing_mm: tuple[float, float]  # (x, y), finite, > 0
    well_size_mm: tuple[float, float]  # (x, y), finite, > 0, below the spacing
    circular_wells: bool
    well_spacing_source: PlateSource
    well_size_source: PlateSource


def load_lab_plates(text: str | None = None) -> dict[str, LabPlate]: ...
def register_lab_plates() -> tuple[str, ...]: ...
```

- `load_lab_plates(None)` reads the packaged file
  (`importlib.resources.files("smc") / PLATES_FILE`, `encoding="utf-8"`);
  `text` is for tests. A bad name, an unknown key or a bad value raises
  `ValueError` naming `plates.toml` and the plate.
- `register_lab_plates()` loads, refuses a `useq` name (D4), registers each
  plate with `rows`, `columns`, `well_spacing`, `well_size` and
  `circular_wells` (not `description`: `useq`'s `name` stays the key), and
  returns the sorted names. Idempotent.
- `smc.plates` imports nothing from `smc` (it runs inside `smc/__init__`).

### 22.5 Not here

- A refusal that names a plate turned 180° in its holder: #414.
- A stand-local plate file, a CLI to list plates, a page view of a plate's
  provenance.
- Changing `SCALE_TOLERANCE`: the guard is right; the definition was wrong.

### 22.6 Not verified (for the owner)

- `plate-walls` at the Ti2-E with `plate=greiner-24-well`, the plate the
  right way in its holder (A1 at the back left, inferred from §22.1): the
  columns run towards +X, the residual, and `goto-well` to a fourth well
  within 100 µm of its centre (§17.7).
- The maker's datasheet: the Greiner reference, read from the plate; its
  pitch against 19.45 mm. A different pitch is a new name (D5).

## 23. A plan's calibration from another run document (#383)

The owner registers the plate with the 4× and scans with the 40×. `scan-plan`
takes its camera matrix from the run document's `camera-calibration` when it
is of the plan's objective (§7.3), and `plate-walls` read the 4×'s from the
same slot. A run document holds one record per plugin (§1.7), so a 40×
calibration run into it replaces the 4×'s: `plate-walls` becomes stale, and
`plate` with it (`plate-walls read its inputs from an earlier run`). Without
it, `scan-plan` sets the 4×'s aside (`camera calibration not used: it is of
<4×>, the plan is for <40×>`) and plans with the profile's size, **rotation 0
and no mirror assumed**. `route` takes the cells' stage positions from the
plan's matrix (§11); the Ti2-E's camera is turned about 177° (session 3,
#405), so a cell at `(dx, dy)` px from a field's centre is routed to about
`(−dx, −dy)`: up to about 270 µm off at the 40×.

**Decision**: `scan-plan` gains a parameter `calibration`, a run document of
its own holding a `camera-calibration` made with the plan's objective in
place, read by the rule `objective-offsets` reads its targets' with
(`calibration_from`, §20.3). The operator keeps one document per objective
(§20, step 1), and `run.json` keeps the registration's.

Set aside:

- **A slot with one record per objective.** The run document's one record
  per plugin (§1.7) is what `provider`, `is_stale`, `unusable`, the runner's
  reads and the page's checklist rest on; a slot keyed by objective changes
  all of them for one slot. It would need an ADR.
- **Staleness that ignores a calibration of another objective.** The runner
  would have to know which objective each reader used, and the slot would
  still hold one calibration: the page names a wall's sides with the
  document's calibration (`PlatePage`, §19), which would then be the
  40×'s for walls the 4× imaged (FM-189).

**Not here**: the plate page (the parameter shows under *Advanced*, as
`camera-calibration`'s `from_run_doc` does; a picker of documents is not
added); `plate-walls` or `goto-well` reading a calibration from another
document; staleness across documents: a `camera-calibration` run again into
the document a plan read does not make the plan stale, as for
`objective-offsets` (§20). The plan records when the calibration was
measured.

### 23.1 What was measured (2026-11-10, planning #383)

On main at `caaa60e` (pymmcore-plus 0.18.1, useq-schema 0.9.2, Python
3.13.0), with a prototype in the design session's scratchpad that replaced
`_camera_basis` by one reading `calibration_from` first:

- `profiles/demo-sample.toml` with `[sample]` as §20.1 (camera 30° and
  mirrored, plate 1.2°), the stage's `Velocity` at 100 000: `camera-calibration`
  under 20X into `cal-20x.json` (0.49950 µm per pixel, 29.967°, mirrored),
  under 40X into `cal-40x.json` (0.24975, 29.967°, mirrored), then under 10X
  (in place at open, 0.99933) and `plate-walls` (96-well, A1, H12, A12, 300 µm
  off) into `run.json`: 1.9 s in all.
- `scan-plan` (`objective` 40X, not turned in; `wells=A1`, 2 × 2) with the
  40X document: `done`, its matrix the 40X calibration's
  (`((−0.21636, 0.12475), (−0.12475, −0.21636))`), `matrix_source`
  `camera-calibration`, `assumed` only the frame's line (the demo-sample
  profile has no `[objectives]`). The same in a dry run (a dry run may read
  a real result, `RunDocument.unusable`). With the 20X document: refused,
  naming both objectives.
- After both plans, `RunDocument.unusable("plate", ...)` is `None`: the
  registration is untouched.
- The runner records every slot of `reads_optional` read at the start
  (`runner.py`, `_read_of`), so a plan with `calibration` still records
  `camera-calibration` of `run.json` as read. That is harmless: a new
  `camera-calibration` in `run.json` makes `plate-walls`, and so the plan,
  stale anyway.
- `calibration_from`'s time is the record's `finished_utc` to the second
  (`2026-11-10T19:50:30+00:00`).

### 23.2 The parameter — `scan_plan/plugin.py`

```python
class ScanPlanParams(PluginParams):  # gains
    calibration: Path | None = Field(
        None,
        description=(
            "A run document holding a camera-calibration made with the plan's "
            "objective in place, for a plate registered with another objective "
            "(relative to the working directory, as from_run_doc). Its matrix "
            "is used, rotation and mirror included."
        ),
    )
```

A second model validator: `pixel_size_um` and `calibration` both given is a
`ValueError`, `give pixel_size_um or calibration, not both`. The field comes
after `pixel_size_um`.

### 23.3 The matrix, in this order

1. `pixel_size_um` given: unchanged.
2. **`calibration` given** (new): `calibration_from(scan-plan, path,
   profile=microscope.profile.microscope.name, dry_run=ctx.dry_run)`, its
   refusals as they are. Then, `where = path.as_posix()` (FM-24, never
   resolved: FM-50), each a `PluginRunError` of `scan-plan` raised outside
   any handler (FM-98):
   - its `objective` is not the plan's: `the calibration in <where> is of
     <was>, the plan is for <now>: give calibration a run document whose
     camera-calibration was made with <now> in place` (`an unknown
     objective` for `None`, on either side, as `_why_not_used`);
   - its `image_shape_px` is not the camera's: `the calibration in <where>
     is of a frame of <its shape>, the camera gives <shape> (binning or the
     region changed): run camera-calibration into it again with this camera
     setting`.

   Otherwise its matrix and size, `matrix_source="camera-calibration"`, and
   in `assumed` only `_gap_line`'s line when there is one, as for (3). The
   document's own `camera-calibration` is not read (`upstream_or_none` is
   not called): the parameter wins, even over one of the same objective.
   The plugin logs at INFO `camera calibration of <objective> from <where>,
   measured <when>`.
3. The document's `camera-calibration` of the plan's objective and frame:
   unchanged.
4. The profile's size: unchanged, except the line of a document's
   calibration set aside because its objective differs (its own unknown
   included), when the plan's objective is not `None`, which gains a hint:
   `camera calibration not used: it is of <was>, the plan is for <now>;
   give calibration, a run document whose camera-calibration was made with
   <now> in place`. The frame-shape line, and a plan for an unknown
   objective (no turret), are unchanged.

A helper `_calibration_given(name, path, ctx, profile, objective, shape) ->
tuple[_Basis, str, str]` (the basis, `where`, `when`) holds (2); `_Basis`
does not change.

### 23.4 The result — `slots/scan_plan.py`

`ScanPlanResult` gains, after `objective_shift_um`:

```python
calibration_from: str | None = None
calibration_measured: str | None = None
```

`calibration_from` is the document the matrix came from, as typed
(`as_posix()`), and `calibration_measured` when it was measured
(`calibration_from`'s second value); both `None` when the matrix is not from
another document, and in every document written before #383. The docstring
says so.

### 23.5 The scan's advice — `scan/plugin.py`

When the camera's size is more than 10 % from a `camera-calibration` plan's,
the refusal's advice (§8, refined while planning #213) names the document
when the plan read one: `run camera-calibration into <calibration_from>,
then scan-plan, again with the stand as it is now`. Without it the text is
unchanged. The advice that sent the operator to `run.json` is the one that
makes the plate stale.

### 23.6 Docs — `docs/plugins/scan-plan.md`

The list of where the matrix comes from gains the parameter as its second
item, with the new hint on the line of item 3; a short example *Registered
at one objective, scanned at another*: the 40× calibrated into a document of
its own (`smc objective set ... --confirm`, `smc run camera-calibration
--run-doc cal-40x.json`), then `smc run scan-plan --run-doc run.json --param
objective=... --param calibration=cal-40x.json`. The parameter's row in the
parameters table.

### 23.7 Tests

- `tests/test_scan_plan_calibration_demo.py` (new; `pytest.mark.demo`): a
  module fixture as §23.1 (the profile and the three documents through
  `run_plugin`, FM-124; `skip` without the demo adapters); each test copies
  `run.json` into its `tmp_path` and passes the calibration documents by
  path.
  - `test_a_plate_registered_at_one_objective_is_planned_with_another_objectives_calibration`,
    parametrized over a real and a dry run: the plan for 40X with
    `cal-40x.json` is `done`; its `matrix_um_per_px` and `pixel_size_um`
    equal the 40X record's (read from `cal-40x.json`); the matrix is not
    `pixel_matrix(pixel_size_um)` and its determinant is negative (turned
    and mirrored: FM-203, FM-73); `matrix_source == "camera-calibration"`;
    `calibration_from` is the path as passed, `as_posix()`;
    `calibration_measured` is the 40X record's `finished_utc` to the second;
    no line of `assumed` holds `rotation 0`; `unusable("plate", ...)` is
    `None` before and after, and `unusable("scan-plan", ...)` after.
  - `test_a_calibration_of_a_third_objective_is_refused_naming_both`: the
    20X document for a 40X plan; the exact text of §23.3, the stage
    `failed`, `__context__` and `__cause__` `None` (FM-98).
- `tests/test_scan_plan_demo.py` gains (its `_seed` and `_write_calibration`
  write a calibration into any document):
  - `test_a_calibration_document_wins_over_the_documents_own` (the same
    objective in both, 0.4 µm in `run.json`, 0.2 µm and another rotation in
    the other: the plan's is the other's);
  - `test_a_calibration_document_of_another_frame_is_refused`;
  - `test_a_calibration_document_without_a_calibration_is_refused_with_its_path`
    (a document holding only a plate);
  - `test_a_dry_run_calibration_document_is_refused_for_a_real_plan`;
  - `test_pixel_size_and_calibration_together_are_refused` (the params
    model);
  - the `another-objective` and `unknown-objective` cases of
    `test_a_calibration_of_another_objective_or_frame_is_ignored_and_says_why`
    gain the hint of §23.3 (4); `transposed-frame` keeps its text.
- `tests/test_scan_demo.py` gains
  `test_a_plan_from_a_calibration_document_names_it_when_the_size_changed`:
  as `test_a_calibrated_plan_is_compared_within_ten_percent`, the
  `scan-plan` record's result rewritten with `calibration_from` set
  (`write_stage` of `StageRecord.model_validate({...})`, §20.1's pattern),
  the advice of §23.5 exact.
- `tests/unit/test_slots.py`: a `ScanPlanResult` dump without the two fields
  loads with `None`.

**Failure modes**: FM-24, FM-50, FM-73, FM-98, FM-124, FM-125, FM-189,
FM-203, and a new entry for the class this section closes: *a per-objective
value kept in a slot that holds one record per document* (the develop
session gives it the next free id).

### 23.8 Not verified (for the owner)

- At the Ti2-E (session 4): a plate registered with `plate-walls` at the 4×
  in `run.json`, a plan for the 40× with `calibration=cal40x\run.json`, a
  scan, cells, and a route whose positions land on cells at the 40×
  (session 3's three calibrations agree on 177.44°, #405, so the matrix
  differs from the 4×'s by its size only).

## 24. The plate's orientation in its holder (#434, #422)

A plate whose wells form a uniform grid lands on itself when it is turned 180°:
the well positions stay and only the names swap (row `r` becomes `R-1-r`, column
`c` becomes `C-1-c`; B3 becomes C4 on a 24-well). The samples follow the names
printed on the plate, so a turned plate that the software does not know about is
imaged at C4 under the name B3. `calibrate` sees the turn when the operator names
wells from the plate, and refuses it as a misnamed well (#414); `plate-finding`
cannot see it at all, and fits a turned plate at 0° with every name swapped and
no word. Only the operator knows which way the plate lies, so **the operator
declares it**: the orientation is how the plate was put down this time, not a
property of the stand (#414's option 2, a stand orientation in the profile, was
dropped for that reason).

This section is the layer's half (#434): the slot keeps the orientation, the
tools that fit a plate take it as a parameter and check or apply it, a pure tool
renames a registered plate without measuring it again, and one rule of the run
document lets that tool read the slot it provides. Every tool that reads `plate`
then names the wells from it with no change of its own. The page's half (the
choice, the map's back and front, the A1 corner, the switch after a
registration) is m5 §16 (#422), against the interface fixed in §24.7.
**Merging this section decides it.**

### 24.1 What was measured (2026-11-11, planning #422)

On main at `fcda44b` (useq-schema 0.9.2, pydantic 2.13, Python 3.13), with
prototypes in the design session's scratchpad:

- **A turn is always a renaming.** For each of the ten registered plates
  (`useq`'s nine and `greiner-24-well`), a plan at A1 (1234, -567) µm turned
  13°, and the same plan turned 180° about its grid's centre (A1 moved to the
  old centre of the last well, rotation + 180°): every well `(r, c)` of one lies
  on well `(R-1-r, C-1-c)` of the other, worst 4.0e-11 µm. A `useq.WellPlate`
  is a uniform grid whatever its pitches and its well shape, so #422's item 7
  ("plates with no 180° symmetry") has no case among the registered plates.
- **Session 3's hand centres** (#414), B3 (10 003, -10 051), B5 (-28 982,
  -9 919), C3 (10 031, 9 375) µm, on `greiner-24-well`: refused by `calibrate`
  at the default ±10° (`the fit turns the plate by 179.8°, beyond ±10°`); with
  the bound lifted, rotation **179.789°**, residual **44.53 µm**, scale error
  +0.07 %, A1 at (48 788, -29 656). The same result turned 180° about the
  grid's centre: rotation -0.211°, refs renamed C4, C2, B4, residual 44.53 µm
  (unchanged), and its C4 lies on the first result's B3 (3.6e-11 µm). Turned
  twice: A1 back to 6.9e-11 µm, rotation to 3e-14°.
- **`wall_position` turns a side with the plate.** On that 179.8° result,
  `wall_position(result, "B3", "+X")` lies at (-7 800, +29) µm from B3's
  centre: on a turned plate `goto-well side=+X` goes to the stage's **-X**
  wall, whereas a side is a stage direction (§19). `centre_from_walls` reads
  stage coordinates and is right on either orientation.
- **Text parameters.** `smc run ... --param turn_deg=180` against a field
  `Literal[0, 180]` is refused (`Input should be 0 or 180`: pydantic's lax mode
  does not turn the text "180" into an int literal); a string `Literal` is
  accepted, as `goto-well`'s `side` is.
- **The run document.** `RunDocument.provider(slot)` is the latest start among
  the records that provide `slot`; a record that read `plate` and provides
  `plate` is then its own provider, differs from the `ReadRecord` it holds, and
  `is_stale` says so forever (reproduced on a synthetic document). The
  registry's check 5 (§1.5) refuses such a plugin today.
- **Who reads a plate's rotation** (every use in `src/`, read for this
  section): `scan-plan` (`to_plan`, `well_geometry`; its in-well grid is
  stage-aligned at 0, 30 and 180°, measured), `route` (`abs(cos)`, `abs(sin)`
  for a square well's box), `density` and `report` (`well_geometry`, a full
  turn), `objectives.plate_for_objective` (a translation), `nearest_well`,
  the page's map (`WellView.rotation_deg`). Wells are selected and ordered by
  their row and column in `useq`'s names (`parse_wells`, `order_wells`), never
  by sorting positions. Only `calibrate`'s bound, the seed's 0°,
  `plate-finding`'s ±4° search, `wall_position` and the page's estimate assume
  a plate that is not turned.

### 24.2 Decisions

**D1. Two values, in the stage's terms.** `PlateOrientation =
Literal["not-turned", "turned-180"]`. `not-turned` is `useq`'s layout on the
stage: A1 at the -X +Y corner, columns towards +X, rows towards -Y (§2.1).
`turned-180` is that plate turned 180°: A1 at the +X -Y corner. The operator's
words ("A1 at the back left") come from the stand's directions on the page (m5
§16) and are never stored: the slot stays the same arithmetic on every stand.
Text values, not `0`/`180`, because a number literal cannot be given on the
command line (§24.1).

**D2. The slot keeps it.** `PlateResult.orientation: PlateOrientation =
"not-turned"`, after `objective`. A run document written before it reads
`not-turned`, which is what every earlier fit was checked against (`calibrate`
refused anything beyond ±45°). A model validator refuses a result whose rotation
is more than 90° from its orientation (`wrap_deg(rotation_deg -
ORIENTATION_DEG[orientation])`), so the two fields never contradict each other.
It is not in the profile.

**D3. The tools that fit take it; `calibrate` checks it.** `plate-registration`,
`plate-walls` and `plate-finding` gain `orientation: PlateOrientation =
"not-turned"`. `calibrate` takes `orientation` and compares the fitted rotation
with it: `delta = wrap_deg(rotation_deg - ORIENTATION_DEG[orientation])`.
Within ±`max_rotation_deg`, the result carries the orientation. Within
±`max_rotation_deg` of the *other* orientation, it is refused with §24.4 (1):
the plate lies the other way, **or two wells are swapped** (two references
exchanged fit at 180° exactly, so the message names both). Anything else keeps
today's misnamed-well refusal, with `delta` in it. `max_rotation_deg` keeps its
meaning (a tilt in the holder), its default (10°) and its cap (45°).

**D4. A one-well seed takes its rotation from the declaration.** One reference
sets A1 with `rotation_deg = ORIENTATION_DEG[orientation]`: A1 = the well's
position minus the nominal offset turned by that angle. Its `assumed` line is
§24.4 (3): the rotation came from the declared orientation, not from a
measurement.

**D5. `plate-finding` cannot see it, and says so.** It fits as it does today
(±4° about `not-turned`) and, when the declaration is `turned-180`, turns its
result (D6). Its `assumed` always holds §24.4 (4), on either orientation:
until now the survey assumed `not-turned` silently.

**D6. The turn is a renaming, done in one place.** `PlateResult.turned()`: A1
moves to the old centre of the last well (row `R-1`, column `C-1`, through
`well_geometry`); `rotation_deg` becomes `wrap_deg(rotation_deg + 180)`;
`orientation` becomes the other one; every well name the result holds goes
through `turned_well` (`refs[].well`, `walls.wells[].well`); `method`,
`residual_um`, `scale_error`, `n_refs`, `fit`, `objective` and the other fields
of `walls` are kept; `assumed` keeps its lines and gains §24.4 (5). Positions
are not recomputed from anything measured: nothing is measured again.
`PlateResult.with_orientation(o)` returns the result itself when it already
has `o`, and `turned()` otherwise. It is **absolute**: asking for the fit's own
orientation gives the fit's numbers back (§24.1: 6.9e-11 µm).

**D7. A pure tool turns a registered plate: `plate-orientation`.**
`requires = ()`, `reads = ("plate",)`, `provides = "plate"`, one required
parameter `orientation`. It reads the latest *other* provider of `plate` (D8)
and returns `with_orientation(params.orientation)`. It is the CLI's way to say
"the plate lies the other way" after a fit without measuring again
(`smc run plate-orientation --param orientation=turned-180`), and the page's
switch runs it (m5 §16; ADR-0006: nothing exists only in the UI). The tools
after it read its result like any `plate`; it makes them stale like any new
`plate`, which is right, since their well names changed. A fit run again later
is the newer provider and the turn is set aside with no extra rule.

**D8. A read never resolves to the reader's own record.** One rule of the run
document, so that D7's tool is not stale by construction:

- `RunDocument.provider(slot, *, excluding: str | None = None)`: the latest
  start among the records that provide `slot`, the record named `excluding` left
  out.
- `is_stale(plugin)` (and `_stale`) compares each slot that `plugin` read with
  `provider(slot, excluding=plugin)`.
- `unusable(slot, *, profile, dry_run, reader: str | None = None)` judges
  `provider(slot, excluding=reader)`.
- The runner's `check_ready`, `_reads_of` and `_read_of`, `RunContext`'s
  `_result_of`, the CLI's `_outdated` and the page's readiness check of a tool's
  `reads` pass the reading plugin's name.
- The registry's check 5 (§1.5) allows `provides` in `reads`; it still refuses it
  in `reads_optional` (an optional read of one's own slot means nothing).

For every plugin that does not read its own slot nothing changes: its record
never provides a slot it reads, so leaving it out leaves the same provider. Every
other reader of `plate` (a consumer, the page's map, `unusable` with no reader)
still sees `plate-orientation` when it is the latest. This is a rule of §1.7, not
an architecture change: no slot, model or file format changes. A record whose
reads are judged without itself cannot be its own input, so `_stale`'s recursion
keeps its `visited` guard and gains no cycle.

**D9. A wall's side stays a stage direction.** `wall_position` turns the
half-size offset by the plate's rotation **from its orientation**,
`wrap_deg(rotation_deg - ORIENTATION_DEG[orientation])`, not by `rotation_deg`:
`+X` is towards the stage's +X on either orientation (FM-189). The texts that
say "`+Y` is towards row A" (`goto_well.py`, `plate_registration/plugin.py`,
`algorithm.py`, `docs/plugins/plate-registration.md`) say "towards row A on a
plate that is not turned".

**D10. What does not change.** `goto-well`, `scan-plan`, `route`, `density`,
`report`, `nis-export`, `objective-offsets` and `objectives.py` read names and
positions through `to_plan` and `well_geometry` (§24.1), so a plate declared
`turned-180` sends `goto-well well=B3` to the printed B3 with no change to them.
`PlateResult.rotation_deg` keeps its bounds ([-180, 180]).

**D11. Not here.** A stand whose stage axes are mirrored as seen from above (no
turn maps `useq`'s layout onto it; none is known); turns of 90° (a plate does not
fit its holder that way); `[sample] rotation_deg`'s ±45° bound (the demo-sample
stand draws a plate that is not turned; the tests declare the orientation, which
is all `plate-finding` can be told); a warning when a declaration disagrees with
a previous run document.

### 24.3 The interface

```python
#: smc/plugins/geometry.py
def wrap_deg(angle_deg: float) -> float:
    """``angle_deg`` brought into (-180, 180]."""
```

```python
#: smc/plugins/slots/plate.py
PlateOrientation = Literal["not-turned", "turned-180"]
ORIENTATION_DEG: Mapping[str, float] = {"not-turned": 0.0, "turned-180": 180.0}


def turned_well(plate: str, well: str) -> str:
    """The name of the well at ``well``'s place once the plate is turned 180°."""
```

```python
class PlateResult(PluginResult):  # gains, after `objective`
    orientation: PlateOrientation = "not-turned"

    def turned(self) -> PlateResult: ...
    def with_orientation(self, orientation: PlateOrientation) -> PlateResult: ...
```

`turned_well` normalises its input (`normalise_well`), finds the row and column
in `useq`'s names (never by parsing letters, §3.2) and raises `ValueError` for a
well that is not on the plate: `turned_well("24-well", "b3") == "C4"`,
`turned_well("96-well", "A1") == "H12"`, `turned_well("1536-well", "A1") ==
"AF48"` (measured).

```python
#: smc/plugins/plate_registration/algorithm.py
def calibrate(
    plate: str,
    refs: Sequence[WellRef],
    *,
    max_rotation_deg: float = MAX_ROTATION_DEG,
    orientation: PlateOrientation = "not-turned",
) -> PlateResult: ...
```

```python
class PlateRegistrationParams(PluginParams):  # gains
    orientation: PlateOrientation = Field(
        "not-turned",
        description=(
            "Which way the plate lies in its holder: not-turned (A1 at the "
            "stage's -X +Y corner) or turned-180 (A1 at the +X -Y corner)."
        ),
    )
```

`PlateWallsParams` and `PlateFindingParams` gain the same field with the same
text; `plate-walls` passes it to `calibrate`, `plate-finding` to
`with_orientation` on its result (D5).

```python
#: smc/plugins/plate_orientation.py (new), entry point `plate-orientation`
class PlateOrientationParams(PluginParams):
    orientation: PlateOrientation = Field(
        description=(
            "Which way the plate lies: not-turned or turned-180. The latest "
            "plate fit is renamed to it; nothing is measured again."
        )
    )
```

```python
class PlateOrientationPlugin:
    name = "plate-orientation"
    summary = "Say which way the registered plate lies; renames its wells."
    requires: tuple[type, ...] = ()
    reads = ("plate",)
    reads_optional: tuple[str, ...] = ()
    provides = "plate"
    Params = PlateOrientationParams
    Result = PlateResult
```

`run` is `ctx.upstream("plate", PlateResult).with_orientation(params.orientation)`
and nothing else: it touches no device and writes no file.

```python
#: smc/run/document.py
class RunDocument:  # changes
    def provider(
        self, slot: str, *, excluding: str | None = None
    ) -> StageRecord | None: ...
    def unusable(
        self,
        slot: str,
        *,
        profile: str,
        dry_run: bool,
        reader: str | None = None,
    ) -> str | None: ...
```

### 24.4 The texts

1. `calibrate`, the other orientation:
   `f"the fit turns the plate by {rotation_deg:.1f}°: it lies turned 180° from
   the declared orientation ({orientation}), or two reference wells are swapped;
   if A1 is at the other corner, declare orientation={other}, else check the
   well names"`.
2. `calibrate`, any other rotation: today's text with `delta` in place of
   `rotation_deg`, and ` from turned-180` after the angle when the declaration is
   `turned-180`: `f"the fit turns the plate by {delta:.1f}°{suffix}, beyond
   ±{max_rotation_deg:g}°: a reference well is misidentified (two references
   the same distance apart fit either way)"`. On `not-turned` it is today's text
   to the character.
3. The seed's `assumed`: `f"one reference well ({name}): rotation {deg:g}°
   assumed from the declared orientation ({orientation}), not measured"`. On
   `not-turned` it still holds `rotation 0° assumed`.
4. `plate-finding`'s `assumed`: `f"orientation {orientation} given by the
   operator: a plate turned 180° lands on itself, so the survey cannot see it"`.
5. A turn's `assumed`: `f"turned 180° to {new} after the fit: every well renamed
   (row r to R-1-r, column c to C-1-c), nothing measured again; a well named in
   a line above is the fit's name"`.

### 24.5 Docs

`docs/plugins/plate-orientation.md` (new; the docs index and the tool table
list it): what it does, when to run it (the cross on the page or a well's
printed label says the plate lies the other way), what it never does (measure),
the CLI line, and that the tools after it are then stale. `plate-registration.md`,
`plate-walls.md` and `plate-finding.md`: the parameter in their tables, the
refusal (1) in place of #414's, and the A1 corner of each value. The "`+Y` is
towards row A" lines say "on a plate that is not turned" (D9).

### 24.6 Tests

FM-73: the tests use a plate whose grid is not square (24-well, 4 × 6, and
96-well) and positions that differ in X and Y, so that a swap of rows and
columns, or a mirror in place of a turn, fails. Values are synthetic from
`useq.WellPlatePlan(..., rotation=180.0 + tilt)` unless they are session 3's.

- `tests/unit/test_plate_orientation.py` (new):
  `test_turned_well_swaps_row_and_column_on_every_registered_plate`
  (every well of every registered plate, against the indices),
  `test_a_turn_keeps_every_well_position_on_every_registered_plate` (≤ 1e-6 µm),
  `test_turning_twice_gives_the_fit_back`,
  `test_a_turn_renames_refs_and_wall_wells_and_keeps_the_residual`,
  `test_with_orientation_of_the_same_orientation_is_the_result_itself`,
  `test_a_result_whose_rotation_contradicts_its_orientation_is_refused`,
  `test_a_document_without_an_orientation_reads_not_turned`,
  `test_wrap_deg_is_in_minus_180_exclusive_to_180_inclusive` (180, -180, 540,
  -180.2, 359.8).
- `tests/unit/test_plate_registration.py` gains:
  `test_session_3_centres_register_when_declared_turned` (§24.1's values,
  rotation within 0.5° of 179.79, residual < 100 µm, `orientation ==
  "turned-180"`),
  `test_session_3_centres_declared_not_turned_are_refused_as_turned` (text (1)),
  `test_a_turned_fit_beyond_the_tilt_bound_keeps_the_misnamed_well_refusal`
  (text (2) with the suffix),
  `test_two_swapped_references_are_refused_with_the_turned_text`,
  `test_a_seed_on_a_turned_plate_places_a1_at_the_far_corner` (text (3)),
  `test_wall_position_plus_x_is_the_stage_plus_x_on_a_turned_plate` (and the
  three other sides, on both orientations).
- `tests/unit/test_run_document.py` gains:
  `test_a_stage_that_reads_its_own_slot_is_judged_on_the_other_provider`,
  `test_a_transform_after_a_fit_is_not_stale_and_its_readers_are`,
  `test_a_fit_run_again_after_the_transform_is_the_provider`,
  `test_a_failed_fit_after_the_transform_leaves_the_slot_unusable`,
  `test_excluding_a_plugin_that_does_not_provide_the_slot_changes_nothing`.
- `tests/unit/test_plugin_registry.py` gains
  `test_a_plugin_may_read_the_slot_it_provides_but_not_optionally`.
- `tests/test_plate_orientation_demo.py` (new; `demo_microscope`):
  `test_plate_orientation_renames_the_registered_plate_through_the_runner`
  (`plate-registration` with refs from a turned 24-well plan declared
  `not-turned` is refused; declared `turned-180` it is `done`; then
  `plate-orientation` to `not-turned` and back: each `done`, `unusable("plate",
  ...)` `None` after each, positions kept, names swapped),
  `test_goto_well_on_a_turned_plate_reaches_the_printed_b3` (the stage at the
  turned plan's B3 within 1 µm, and not at the not-turned plan's B3),
  `test_goto_well_side_plus_x_moves_towards_plus_x_on_a_turned_plate`,
  `test_scan_plan_after_a_turn_is_stale_and_plans_the_turned_names`.
- `tests/test_plate_finding_demo.py` gains
  `test_plate_finding_names_wells_from_the_declared_orientation`
  (the module's `_on_demo` with the `PlateSample` of
  `test_the_survey_fills_the_plate_slot_on_the_demo`, run once per
  orientation: the `turned-180` result's A1 lies on the `not-turned` result's
  last well within 1 µm, and `assumed` holds text (4) on both).

**Failure modes**: FM-73 (symmetric data), FM-189 (a side named in one frame
and applied in another, D9), FM-217 (an estimate drawn like a measurement: an
orientation that was given is said to be given), FM-165 (the label belongs to the
result, whichever branch), FM-111 (a read recorded against the provider it
used), and a new entry named by its text for the develop session to number: *a
read that resolves to the reader's own record* (a tool that reads the slot it
provides is stale forever, or reads its own previous output, unless the read
leaves its own record out).

### 24.7 For the page (#422, m5 §16)

- The page fills `orientation` itself: `PAGE_PARAMS` gains it for
  `plate-registration`, `plate-walls`, `plate-finding` and `plate-orientation`.
- The switch after a registration is `plate-orientation` through `run`, the
  path of every tool.
- Captures and a selection are names: `turned_well` renames them.
- The estimate (m5 §15 D2) uses `ORIENTATION_DEG` about the holder's centre.

### 24.8 Not verified (for the owner)

- At the Ti2-E: the Greiner plate turned, declared `turned-180`, registered
  from three wells named from the plate, and `goto-well` to a named well checked
  by eye against the printed label; then the same with the plate not turned.
- That the plate holder takes the plate either way round (#414 option 1 was
  inferred, not tried).
