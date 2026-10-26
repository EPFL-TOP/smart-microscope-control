# M3 design — plate tools as plugins

- **Status**: design for issue #18 (§1, 2026-10-26).
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
and the run document (#18), plate registration (#19), automatic plate
finding (#20), wells and the field grid (#21), camera↔stage calibration
(#22), object detection (#23) and throughput timing (#24). §1 fixes #18;
each of the others gets its section when its plan is written, and may
extend §1 only additively (a new optional `RunContext` method, a new
optional record field with a schema bump, §1.7).

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
   `reads_optional` not disjoint, or holding `provides`.
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
than the older one silently.

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
  bump); §1 does not preclude either.
- **Several writers of one run document** (a UI and a CLI on one
  experiment): `write_stage` re-reads before it writes, and nothing locks
  the file. A lock comes with the UI (M5) if it is needed.
- **Detector plugins** (#23) are a second family behind their own
  protocol; their discovery is #23's.
- **A pipeline command** that runs several stages in order is not in M3:
  each stage is run alone, and the document says what is missing.
- **JSON output** of `smc run` beyond the result (M1 §9: later).

## 2. Sequencing

Issue #18 comes first: every other M3 issue depends on §1. #19 and #20 share the `plate`
slot: #19 defines its model, and #20 is planned after #19's section is
written. #21 reads `plate`; #22 and #24 are independent of the plate and
can run in parallel with #19 once #18 is merged.
