# M3 design — plate tools as plugins

- **Status**: design for issue #18 (§1, 2026-10-26); the plate-scan
  workflow, §2 to §16, for issue #192 (2026-10-27).
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
| `scan-plan` | `scan-plan` | `plate` | `camera-calibration`, `timings` | `Camera` | #21 | 7 |
| `scan` | `scan` | `scan-plan` | — | `XYStage`, `Camera` | #21 | 8 |
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
    side: Literal["left", "right", "top", "bottom"]
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
    method: Literal["seed", "references", "lattice-fit"]
    residual_um: float | None = Field(ge=0.0)
    scale_error: float | None
    n_refs: int = Field(ge=0)
    refs: tuple[WellRef, ...] = ()
    fit: LatticeFit | None = None

    def to_plan(self, **kwargs: Any) -> useq.WellPlatePlan: ...
    def well_center_um(self, well: str) -> tuple[float, float]: ...
```

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
when the measurement was reused.

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
  from the start of the attempt that took the frame.
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
  documents are attached to issues); `model_sha256` its SHA-256;
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
  is not 2-D is a `ValueError` (the layer's `Camera.snap` is 2-D).
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
  division.

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

**Reuse.** With `from_run_doc`, nothing moves: the other document is
loaded (`RunDocument.load`, its errors become `PluginRunError`), its
`provider("camera-calibration")` must be `done`, of the same profile, and
of the objective in place (when there is a turret), else `PluginRunError`
naming what differs. The result is that one with `reused_from` set to the
path's POSIX text and `return_error_um=None`, and `assumed` gaining
`camera calibration reused from <path>, measured <finished_utc>`.

**Measurement**, refused in a dry run (`PluginRunError`: "a dry run cannot
measure the camera; run it without --dry-run"):

1. The step: `step_um`, else `DEFAULT_STEP_FRACTION × width × pixel size`
   when `Camera.pixel_size_um()` is finite and positive, else
   `FALLBACK_STEP_UM` with `assumed` gaining `step of 50 µm: no pixel size
   was known` (a probe distance, not a result).
2. `start = xy.position_um()`; `reference = camera.snap()`.
3. For `(dx, dy)` in `(step, 0)`, `(0, step)`: `ctx.raise_if_stopped()`,
   `xy.move_to_um(start + (dx, dy))`, `ctx.wait_s(settle_ms / 1000)`,
   `snap`, `measure_shift(reference, frame)`; `ctx.progress`.
4. `fit(...)`; notes → `PluginRunError` with the fit's numbers and notes;
   nothing fills the slot.
5. Only now (FM-88): `move_to_um(start)`, `snap`, and `return_error_um` is
   the stage length (`M` applied) of `measure_shift(reference, frame)`. A
   return error above `max(2.0, 4 × pixel_size_um)` µm adds to `assumed`
   `the stage came back <n> µm off: its repeatability limits the scan's
   stitching` (reported, not refused).

### 4.3 Tests

Ported: `test_camera_calibration.py` (unit) with `test_a_known_shift_is_recovered`,
`test_confidence_is_measured_after_alignment`, `test_a_blank_field_scores_zero`,
`test_pixel_size_and_handedness_are_recovered` (the synthetic convention:
upright is not mirrored), `test_rotation_is_recovered`,
`test_a_stage_that_does_not_move_is_caught`. New, on
`demo_microscope_with_sample` with a textured `PlateSample`:
`test_calibration_recovers_a_rotated_mirrored_camera` (a 384 × 512 frame,
30°, mirrored, pixel size within 2 %, rotation within 0.5°);
`test_a_featureless_field_fails_and_fills_no_slot`;
`test_a_stop_leaves_the_stage_where_it_is` (no return move in the log);
`test_reuse_copies_a_done_calibration_and_says_so`;
`test_reuse_refuses_another_profile_or_objective`.

**Failure modes**: FM-88, FM-64, FM-73, FM-75 (casts), FM-104 (finite
pixel size), FM-85 (readbacks after the facade's move only).

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

Wall touches go the same way: `goto-well --param side=left`, then a
`WallTouch` per wall in `walls`. The plate page (§15) collects the
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
    side: Literal["centre", "left", "right", "top", "bottom"] = "centre"
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
test (plate type, A1, rotation within ±20°, 1 to 4 references with ±12 µm
of jitter); `test_a_seed_fills_the_plate_slot_with_rotation_zero`;
`test_a_residual_beyond_300_um_fails_and_names_the_well`;
`test_a_well_given_twice_is_refused`; `test_goto_well_drives_to_the_wall_point_on_a_rotated_plate`
(demo); `test_goto_well_needs_a_plate`. `docs/plugins/plate-registration.md`
documents §5.1's loop.

**Failure modes**: FM-73 (anisotropic probes, a rotated plate),
FM-43.

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
    seed: int = 0
    statistic: Literal["mean", "std"] = "mean"
```

`plate` sets the survey's extent; `plates` the candidates fitted (empty:
every registered `-well` plate). The centre is both coordinates or
neither (a model validator); neither is the stage's position.

- **The autofocus.** It cannot lock over the plastic between wells. When
  it is engaged, the plugin calls `disengage()` (a safe call, no motion)
  before the first move, never engages it again (FM-88: engaging drives
  Z), and adds a note `PFS was switched off for the survey; engage it
  again before the scan`.
- **The loop**: per point `ctx.raise_if_stopped()`, `move_to_um`, `snap`,
  the statistic, `ctx.progress`. A `SafetyRefusedError` (a point outside
  the soft limits) and any other `HardwareError` except the halt and the
  motion stop skip the point, counted; more than 10 % skipped is a
  `PluginRunError`. The survey is the checkpoint
  (`SurveyCheckpoint(points: tuple[SurveyPoint, ...])`, saved every 50
  points and at the end), so a failed fit keeps it and `--resume` refits
  without moving (§1.4).
- `decide` returning `None` is a `PluginRunError` with its notes; a
  "barely beats" note goes into `fit.notes` of a usable result. A dry run
  is refused as in §4.

**Tests**: the ported `test_plate_finding.py` (synthetic surveys); new
`test_the_survey_fills_the_plate_slot_on_the_demo` with a `PlateSample`
and a coarse step; `test_an_engaged_autofocus_is_switched_off_and_not_relocked`
(fake); `test_a_failed_fit_keeps_the_survey_for_a_resume`. **Failure
modes**: FM-64, FM-88, FM-12.

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
  a `PluginRunError`.
- **The matrix**, in this order: (1) `pixel_size_um` given:
  `pixel_matrix`, `matrix_source="parameter"`, `assumed` gaining `pixel
  size given as a parameter; rotation 0 and no mirror assumed`; (2) a
  usable `camera-calibration` whose `objective` equals the plan's and
  whose `image_shape_px` equals `camera.image_shape()`: its matrix,
  `"camera-calibration"` (a calibration of another objective or shape is
  ignored, and `assumed` says why); (3) the profile's size for the
  objective, `turret.pixel_size_um(objective)`, or with no turret
  `camera.pixel_size_um()`, when finite and positive (FM-104):
  `pixel_matrix`, `"profile"`, `assumed` gaining `no camera calibration
  for <objective>: rotation 0 and no mirror assumed`; (4) otherwise
  `PluginRunError`: "no pixel size for <objective>: run
  camera-calibration with it in place, or add it to the profile's [camera]
  pixel_size_um" (#21's "refuses to plan without a pixel size").
- `image_shape_px` is `camera.image_shape()`; `fov_um` the footprint's
  extent at the origin. When the camera's rotation makes the
  axis-aligned grid open gaps (`sin |θ| > overlap / 2`), `assumed` says so
  and suggests the overlap to use.
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

## 8. The scan (#21)

`smc/plugins/scan/plugin.py`, entry point `scan`. A port of
`scan.run_iter`, redesigned around the layer.

```python
class ScanParams(PluginParams):
    autofocus: Literal["if-engaged", "required", "off"] = "if-engaged"
    max_failed_in_row: int = Field(3, ge=1, le=100)
    max_unlocked_in_row: int = Field(5, ge=1, le=1000)
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

**Per well**, before its first field: the objective (when checked) and the
pixel size are read again; a change of objective, or a pixel size more
than 1 % from `p0` (the magnifier lever, FM-103), is a `PluginRunError`,
the checkpoint keeping the wells done.

**Per field**: `ctx.raise_if_stopped()`; `readback = xy.move_to_um(x, y)`;
when the autofocus is used and not `is_locked()`, one
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
`OSError` writing a frame ends the scan as it is (a full disk is not a
field's fault).

**Checkpoint** after each well (`ctx.save_checkpoint`); a resume skips
`wells_done`, keeps the token and the frames. **Dry run**: no frame is
taken and nothing is written; the moves go through the layer's dry run,
and the result has no frames and `assumed` gaining `dry run: <n> fields
walked, no frame taken`.

The scan never moves Z, never turns the turret, never engages an
autofocus the operator left off, and returns to no start position.

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

## 9. Cells (#23)

`smc/plugins/cells/detector.py`, `algorithm.py`, `plugin.py`, entry point
`cells`. The detector is a port of `detector.py`; the de-duplication
across fields is new.

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

    def detect(self, frame: npt.NDArray[Any]) -> list[Detection]: ...
```

A protocol, and no discovery in M3: one detector, `TorchDetector`, and
the tests' stubs.

```python
DEFAULT_MODEL = "cell_classes_model-cat-v2.pth"


class TorchDetector:
    def __init__(
        self, weights: Path, *, device: str, score_threshold: float
    ) -> None: ...


def default_model_path() -> Path: ...
def load_detector(
    model_path: Path | None, *, device: str, score_threshold: float
) -> Detector: ...
```

- `TorchDetector` imports torch and torchvision **inside** `__init__`
  (the plugin module imports without them, so `smc run --list` shows
  `cells` on every PC). It reads the checkpoint with
  `torch.load(weights, map_location="cpu", weights_only=True)`, takes
  `model_state_dict`, and refuses (`ValueError`) a checkpoint without
  `classes` or whose `classes[0]` is not `__background__`, or whose
  `roi_heads.box_predictor.cls_score.weight` has another number of rows
  than `len(classes)`. It builds
  `fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
  num_classes=len(classes))`, loads the state with `strict=True`, `eval()`.
- `device`: `auto` is `cuda` when available, else `cpu` (never `mps`
  unasked: the port's caution, though MPS ran in §2.1); an unavailable
  device asked for is a `ValueError`.
- `detect`: `normalize_plane` (float32, clipped to the 0.5 and 99.5
  percentiles, scaled to [0, 1]), the plane repeated to 3 channels, one
  frame per forward pass under `torch.no_grad()`, the model's own resize
  kept; boxes from torchvision's `[x0, y0, x1, y1]` to `Detection`;
  `score < score_threshold` dropped; `label = classes[label_id]`.
- `default_model_path()`: `$SMC_MODELS / DEFAULT_MODEL` when the variable
  is set (not empty, FM-78), else `<checkout>/local/models/DEFAULT_MODEL`
  with the checkout `Path(smc.__file__).resolve().parents[2]` (the
  editable install of every PC).
- `load_detector` raises `PluginRunError` for a missing torch ("install
  the detect extra: pip install -e .[detect]"), a missing file (naming
  it, and `SMC_MODELS`), and a checkpoint refused above.

### 9.2 Algorithm

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
def merge_across_fields(
    cells: Sequence[Cell],
    *,
    frames: Sequence[ScanFrame],
    image_shape_px: tuple[int, int],
    radius_um: float,
) -> tuple[tuple[Cell, ...], int]: ...
```

- `suppress_in_frame` is `overlap.select_survivors` with `bbox_iomin`
  (intersection over the smaller box) and `prefer="score"`: the model's
  per-class NMS returns one object as both `single` and `doublet`, and the
  more confident label wins. The 17 tests of `test_overlap.py` come with
  it.
- `to_cells`: the box centre through `stage_of_pixel` (§3.1).
- **`merge_across_fields`** (new): within one well, cells from
  **different** frames whose centres are within `radius_um` belong to one
  object (single-linkage, union-find over a grid hash of cell `radius_um`,
  any label); the one kept is the detection whose box centre is farthest
  from its own frame's edge in pixels (the least cut view), ties broken by
  score then by frame index. Two cells of the same frame are never merged
  (the in-frame step did that). It returns the cells in input order and
  the number removed. A cell that only one frame saw is kept as it is.
  Ownership by nearest frame centre was rejected: with a calibration error
  near the boundary, both frames can disown the same cell.

### 9.3 Plugin

```python
class CellsParams(PluginParams):
    model_path: Path | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    score_threshold: float = Field(0.5, ge=0.05, le=1.0)
    max_overlap: float = Field(0.7, gt=0.0, le=1.0)
    merge_radius_um: float = Field(5.0, ge=0.0, le=100.0)
```

`requires = ()`, `reads = ("scan",)`, `provides = "cells"`. The plugin
gets its detector through the module-level `load_detector` (tests replace
it with `monkeypatch.setattr` on `smc.plugins.cells.plugin`). Per well:
each frame through `load_frame` (a missing frame file is a
`PluginRunError` naming it), `detect`, `suppress_in_frame`, `to_cells`;
then `merge_across_fields` on the well; `ctx.progress` per frame. Cells of
the wells done go to `cells/<token>.partial.json` after each well, and the
checkpoint `CellsCheckpoint(token: str, wells_done: tuple[str, ...])`
follows; the end writes `cells/<token>.json` and removes the partial file.
`seconds_per_frame` is the mean wall time of `detect` (`None` with no
frame). A scan with no frame (a dry run) is a `PluginRunError`.

`merge_radius_um = 5.0` is **chosen, not measured**: half the size of a
yeast cell, well above the calibration's error, well below two cells
side by side. #198 looks at merged pairs on real frames.

**Tests** (stub detector unless said):
`test_a_cell_seen_by_two_overlapping_fields_is_counted_once`;
`test_the_least_cut_view_is_kept`; `test_two_cells_of_one_frame_are_never_merged`;
`test_merge_keeps_cells_of_different_wells_apart`;
`test_a_rotated_mirrored_matrix_puts_cells_at_their_sample_position` (a
`PlateSample` with blobs, a thresholding stub, FM-73);
`test_the_plugin_imports_without_torch` (a subprocess with
`sys.modules["torch"] = None`); `test_a_missing_model_says_where_it_looked`;
`test_a_resume_skips_the_wells_done`;
`test_the_trained_model_loads_and_finds_nothing_on_noise` skipped
without torch or the weights (never in CI). **Failure modes**: FM-73,
FM-78, FM-50 (no full path), FM-113, FM-24.

## 10. Density (#193)

`smc/plugins/density/algorithm.py` and `plugin.py`, entry point
`density`. New, pure.

```python
class DensityParams(PluginParams):
    area_grid_um: float = Field(5.0, ge=0.5, le=100.0)
```

`requires = ()`, `reads = ("scan-plan", "scan", "cells")`.

```python
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
    labels: Sequence[str],
) -> tuple[DensityWell, ...]: ...
```

- The area is the union of the footprints of the well's **taken** frames
  (failed fields excluded), intersected with the well shape, rasterised
  on a grid of `grid_um` (a 6.4 mm well at 5 µm is 1.6 M cells, counted
  with numpy in chunks bounded at 2²⁴ elements, FM-74). The error at 5 µm
  is under 1 % of a field.
- A cell is counted when `well.contains(x, y)`; one outside its well's
  shape (a field corner on the plastic) counts in `outside_well`. Every
  label of `CellsResult.classes` appears in `counts`, zero included;
  `per_mm2` is `None` for an area of 0.

**Tests**: `test_a_full_disc_scan_measures_the_well_area` (within 1 % of
π r²); `test_overlapping_fields_are_not_counted_twice_in_the_area`;
`test_failed_fields_reduce_the_area`; `test_a_cell_on_the_plastic_counts_outside`;
`test_a_well_with_no_frame_has_unknown_density`;
`test_a_grid_too_fine_for_the_well_is_refused_before_allocating` (FM-74).

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
  Counting is numpy over cells × candidates in chunks (FM-74).
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
  frame of its well; `name` is `<well>_r<k:03d>`; `counts` holds every
  label inside the footprint. `travel_s` is the sum of `timings.move_ms`
  over the route when `timings` is usable, else `None`.

**Tests** (hypothesis where marked): `test_no_field_leaves_its_well_or_enters_an_exclusion`
(hypothesis over margins, exclusions, rotations); `test_chosen_fields_never_overlap`;
`test_the_greedy_takes_the_densest_field_first`;
`test_the_route_visits_every_field_once` (hypothesis);
`test_two_opt_never_lengthens_the_route` (hypothesis, against the
nearest-neighbour path); `test_wells_follow_the_scan_order`;
`test_another_objective_scales_the_field`; `test_travel_time_is_none_without_timings`.
**Failure modes**: FM-73, FM-74.

## 12. Timing (#24)

`smc/plugins/timing/algorithm.py` and `plugin.py`, entry point `timing`,
slot `timings`. A port of `timing.py` with its 19 tests; `Budget` and
`plan()` stay in `algorithm.py`.

```python
class TimingParams(PluginParams):
    repeats: int = Field(3, ge=1, le=20)
    distances_um: tuple[float, float] = (200.0, 9000.0)
    with_autofocus: bool = True
    channels: tuple[str, ...] = ()
```

`requires = (XYStage, Camera)`. Stage: from the start, `+d` in X and back,
`repeats` times per distance, the median of each, a line fit clamped at 0
(`_fit_line`). Autofocus (when present, engaged, and `with_autofocus`):
`repeats` × (`disengage()`, timed `engage()`); a sample that does not
lock makes `refocus_ms` the assumed 400 ms with a note, as the port's
regression test pins. `snap_ms`: the median of `repeats` bare snaps.
Channels, when given: the port's switch and snap timings, the channel
found restored on the normal path. The return to the start and the
restored channel run only after a normal end (FM-88). A dry run is
refused. **Failure modes**: FM-88, FM-64, FM-12.

## 13. The static report (#195)

`smc/plugins/report/render.py` and `plugin.py`, entry point `report`.

```python
class ReportParams(PluginParams):
    max_cells_drawn_per_well: int = Field(5000, ge=0, le=100_000)
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
    assumed: Sequence[str],
    max_cells_drawn_per_well: int,
) -> str: ...
```

**One HTML file**, `report/<token>.html`, written UTF-8 through
`write_atomically`, with `<meta charset="utf-8">`, inline SVG and CSS,
**no script and no external URL** (it opens offline on the microscope PC
and attached to an issue). It shows:

1. A header: the run document's file name, the profile, the plate, the
   objective, the date, and every `assumed` line of the upstream results,
   labelled by stage.
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

Every text from a result goes through `html.escape`. **Tests**:
`test_the_report_is_one_file_with_no_script_and_no_url`;
`test_every_scanned_well_appears_with_its_density`;
`test_a_well_name_with_markup_is_escaped`;
`test_the_report_without_a_route_says_so`;
`test_the_cells_drawn_are_capped_and_the_caption_says_so`.

## 14. The NIS-Elements export (#196) — interface only

`smc/plugins/nis_export/format.py` and `plugin.py`, entry point
`nis-export`. **The format is not designed here**: it waits for a real XY
position list exported by NIS-Elements 6.20 on the Ti2-E (#196), and is
not to be guessed.

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
`pfs_offset` the format carries, once known. **Acceptance** is #196's: the
sample's structure round-trips (`decode(encode(x)) == x` and the sample
decodes), and the owner loads one export into NIS on the stand.

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
Without the `ui` extra it exits 1 with "install the ui extra: pip install
-e .[ui]".

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
  to** is `goto-well`.
- `stop()` sets the stop event, then `microscope.stop()` (the emergency
  stop of M1 §13), in that order.
- `view()` is a pure snapshot built from a fresh `RunDocument.load_or_new`
  and the stand's state: every well (`well_geometry` of the `plate` slot,
  or of the nominal plate before one exists), the selection, the density
  per label when `density` is usable, the route's points when `route` is,
  the stage position, the objective, the autofocus state, and the
  checklist of the slots of §2.2 with `unusable`'s reason for each.

### 15.2 The page

One page, ADR-0006's layout: the status strip (XY, objective, PFS, light,
**STOP** always visible); the plate map, a Bokeh figure in stage µm with
Y up (wells as circles or squares, selection filled, density as colour by
a chosen label, the route as a line, the stage as a cross, names); click
modes *select* (default) and *drive* (`goto-well`, an explicit toggle,
`nikon-control`'s "armed mode"); the tools in workflow order, each with
its key parameters and an *advanced* fold from `describe_params`, a Run
button and the progress; the run document as a checklist. A periodic
callback every 250 ms drains the progress queue and refreshes the view on
the server's thread (Bokeh models are not touched from the worker). On
session end: `stop()` if a run is going, then nothing else; the stand
closes with the server. No turret control, no Z control: the page offers
none in M3.

**Tests**: the controller on `fake_microscope` and the demo, with stub
plugins through `EntryPoint`s (§1.12): `test_a_click_and_a_typed_range_build_the_same_selection`;
`test_run_goes_through_the_same_parameters_as_the_cli`;
`test_a_second_run_while_one_runs_is_refused`; `test_stop_sets_the_event_then_stops_the_stand`;
`test_captured_references_reach_plate_registration`;
`test_the_view_shows_why_a_stage_is_unusable`. The page itself:
`test_the_page_builds_on_the_demo` with `pytest.importorskip("panel")`;
CI installs `.[dev,ui]` (#197 edits the workflow). **Failure modes**:
FM-98, FM-70 (the stop path takes no lock the worker holds), FM-44.

## 16. Sequencing

1. **#199, the foundation (§3)**, first and alone: every other issue
   imports it.
2. Then, in parallel, each in its own package (no two issues touch the
   same module): **#22** (§4), **#19** (§5), **#21** (§7, §8), **#23**
   (§9), **#24** (§12), **#193** (§10), **#194** (§11), **#20** (§6).
   Shared files: each adds its entry point line to `pyproject.toml`
   (sorted by name; the rebase conflict is expected and trivial), #22 adds
   `scikit-image`, #23 the `detect` extra, #197 the `ui` extra.
3. **#195** (§13) and **#197** (§15) once the plugins they show exist; both
   build on stubs and can start earlier.
4. **#196** (§14) once the NIS sample is in `local/`.
5. **#198**, session 2 at the stand.

For session 2 the critical path is #199 → #22, #19, #21, #23 → #194 →
#195 or #197 → #196.

### 16.1 Not verified (for #198 and the owner)

- The Ti2-E's pixel size at 40× (0.275 µm assumed) and the camera's
  rotation; the stage's repeatability (§4's return check measures it).
- Whether the PFS holds its lock across a 96-well well at 40× in
  brightfield, and how often `engage()` is needed per field.
- The model's behaviour on Prime 95B frames: it was trained on other
  frames (1024², of a pixel size not recorded in the checkpoint) and
  resizes everything to 800 px, so a 1412² frame is seen at about
  0.49 µm per pixel. Detections are checked by eye on the first real scan.
- Inference time on the microscope PC's CPU (0.34 s per frame on an M5
  Pro); a well of 305 frames would take about 2 minutes at that speed.
- `merge_radius_um = 5.0` and the overlap of 10 % against real cells.
- Whether the stage's move time follows the Euclidean distance (§11).
- The NIS-Elements format (§14).
- Panel's version and behaviour on the microscope PC over RDP.
