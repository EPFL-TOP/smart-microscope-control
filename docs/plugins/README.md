# Plugins

A plugin is an interchangeable microscope tool: plate registration, plate
finding, camera calibration, a scan. It asks the stand for **capabilities**, never
for a device, so the same plugin runs on every microscope that offers them
([ADR-0003](../adr/0003-capabilities-roles-and-profiles.md),
[ADR-0004](../adr/0004-plugin-architecture.md)). The contract (names, rules,
messages) is [`docs/design/m3-plugins.md`](../design/m3-plugins.md) §1; this page
is how to use it.

- [Running a plugin: `smc run`](#running-a-plugin-smc-run)
- [Writing a plugin](#writing-a-plugin)
- [The run document](#the-run-document)
- [Testing a plugin](#testing-a-plugin)

Plugins with a page of their own: [`camera-calibration`](camera-calibration.md)
(how the camera's pixels map to the stage: pixel size, rotation, mirror) and
[`cells`](cells.md) (single cells and doublets on a scan's frames, in stage µm,
each counted once; needs the `detect` extra and the trained weights).

## Running a plugin: `smc run`

```text
smc run --list
smc run NAME --show-params
smc run NAME [-p PROFILE] --run-doc PATH [--param KEY=VALUE]... [--dry-run] [--resume]
```

`--list` and `--show-params` open no stand. A run needs `--run-doc`: the JSON file
that holds the stages of one experiment. It has no default, so a file is never
created where you did not say.

```text
$ smc run --list
stage-report  Read where the stage is, and send no move.  requires XYStage  provides stage-report  reads -

$ smc run stage-report --show-params
samples      int    1     ge=1 le=100  How many readings to take.
interval_ms  float  0.0   ge=0.0 le=10000.0  Wait between two readings.
include_z    bool   True  Also read Z when there is a focus stage.

$ smc run stage-report -p demo --run-doc run.json --param samples=3
stage-report: 1/3 read
stage-report: 3/3 read
✓ stage-report done in 0.0 s — run.json, stage stage-report
{
  "assumed": [],
  "profile": "demo",
  "readings": [
    {
      "x_um": -0.0,
      "y_um": -0.0,
      "z_um": 0.0
    },
    ...
  ]
}
```

(The demo stage sits at the origin and reads `-0.0`; the reading is recorded as the
device gave it.) `stage-report` only reads positions, which is why it is the
built-in plugin: it exercises the whole path on the demo devices without a move.

**Parameters.** `--param key=value`, repeated. A scalar (`int`, `float`, `bool`,
`str`, a path, an enum member, a `Literal`) is given as text; a list or a tuple
takes JSON (`--param 'wells=["A1","B2"]'`); `null` is "none" for a field that
admits it. Every problem is reported at once, and `--show-params` lists the keys.
The quoting of JSON is the shell's business: bash keeps the double quotes inside
single quotes, but PowerShell 5.1 and `cmd.exe` strip them (documented behaviour of
their native-argument passing; check it at the stand), and a value then fails with
`'wells' takes JSON`. Escape each inner quote with a backslash there:
`--param "wells=[\"A1\",\"B2\"]"`.

**What is checked before the stand opens.** The plugin loads, the parameters are
valid, `--run-doc` is a file in a folder that exists, the run document is readable,
and every stage the plugin reads is in it and current. Then the stand opens, and
the capabilities the plugin requires are checked before anything is written: a
stand that lacks the XY stage gets one error naming every missing capability and
`[roles.assign]`, and an untouched run document.

**Stopping.** Ctrl-C stops any move under way, records the stage as `stopped`
(its checkpoint kept) and exits 130. `--resume` continues the last attempt from
its checkpoint, when the parameters and the profile are the same. A real run does
not resume a dry run's checkpoint: it stands for work that was never done. A stage
that is stale (something it read has been produced again since) is not resumed
either: its checkpoint was made against inputs that no longer match.

**Exit codes.**

| Code | Meaning |
|---|---|
| 0 | Done, or a listing. |
| 1 | A hardware, profile or file error; a plugin that does not load; a stand that lacks a capability; a run document that cannot be read or written; a plugin's own exception. When only the final write failed, the result is still printed. |
| 2 | Nothing was sent, and a command fixes it: an unknown plugin, a bad `--param`, a missing or bad `--run-doc`, a stage that is not usable, nothing to resume, a safety refusal. |
| 130 | Ctrl-C, or a plugin that stopped itself. |

When a run makes other stages stale (their input was produced again), one
`! <plugin> read <slot> from an earlier run; run it again` line says so. Nothing
is deleted or refused: running the stale stage again is the operator's call.

## Writing a plugin

A plugin is a class with these members; the registry checks all of them before it
lets the plugin be listed or run.

```python
class StageReport:
    name = "stage-report"
    summary = "Read where the stage is, and send no move."
    Params = StageReportParams
    Result = StageReportResult
    requires: tuple[type, ...] = (XYStage,)
    provides = "stage-report"
    reads: tuple[str, ...] = ()
    reads_optional: tuple[str, ...] = ()

    def run(self, microscope, params, ctx): ...
```

| Member | Rule |
|---|---|
| `name` | Kebab-case (`[a-z][a-z0-9]*(-[a-z0-9]+)*`), equal to the entry point's name. It keys the plugin's record in the run document and names its logger. |
| `summary` | One line, for `smc run --list`. |
| `Params`, `Result` | Subclasses of `PluginParams` and `PluginResult`. |
| `requires` | Capability Protocols (`XYStage`, `ZStage`, `Camera`, ...): a tuple of classes that `smc.hardware.microscope.CAPABILITY_ROLES` knows. |
| `provides` | The slot of the run document the result fills, by the same pattern. By convention the plugin's name, unless several plugins produce the same thing (plate registration and plate finding both provide `plate`). |
| `reads` | Slots the plugin needs; the runner refuses to start without them. |
| `reads_optional` | Slots the plugin uses when they are usable (timings, a calibration). Disjoint from `reads`, and neither holds `provides`. |

`run(microscope, params, ctx)` is the hardware loop: it asks `microscope` for
capabilities (`microscope.require(XYStage)`), does its work, and returns its
`Result`. It never writes the run document itself, and never imports
`pymmcore_plus` or names a device label ([ADR-0003](../adr/0003-capabilities-roles-and-profiles.md)).
The algorithm it applies is a pure function in another module that takes no
`Microscope`, unit-tested on synthetic data
([ADR-0004](../adr/0004-plugin-architecture.md)).

### Parameters and results

```python
class StageReportParams(PluginParams):
    samples: int = Field(1, ge=1, le=100, description="How many readings to take.")
```

The bases make every float finite (`nan` and `inf` are refused even where you
forgot a bound), refuse unknown keys and freeze the model. Put the bounds and a
`description` on every field: `--show-params` prints them. Unknown is `None`,
never `nan` ("unknown beats guessed"). `PluginResult.assumed` lists what the
result rests on that was assumed and not measured. A result that says where to
image is a `useq` object.

### The context

`ctx` is what the plugin gets besides the microscope. It is built by the runner;
a plugin never builds one.

| Member | Use |
|---|---|
| `ctx.progress(done, total=None, message="")` | Report progress. Never validated, never fatal: a callback that raises is dropped after one warning. |
| `ctx.should_stop()`, `ctx.raise_if_stopped()` | Has a stop been asked for? Check between units of work; `raise_if_stopped` raises `StopRequested`, which is an outcome and not an error. |
| `ctx.wait_s(seconds)` | Wait, ending at a stop. **Never `time.sleep`**: it cannot be stopped, and on Windows one long wait keeps Ctrl-C from being delivered. |
| `ctx.upstream(slot, Model)` | The result of the provider of a slot in `reads`, validated by `Model`. |
| `ctx.upstream_or_none(slot, Model)` | The same for a slot in `reads_optional`; `None` when it was not usable. |
| `ctx.save_checkpoint(state)`, `ctx.checkpoint(Model)` | Save what you would not want to repeat after each unit (a well, a field); read the one an attempt resumed from. |
| `ctx.logger` | `smc.plugins.<name>`. Log through it; `smc -v` shows it. A plugin never prints. |
| `ctx.run_dir` | The folder of the run document. Write files under it and store their paths relative to it, as `Path.as_posix()` strings (a Windows `str(Path)` has backslashes, which another OS does not read as a separator), so the folder moves as a whole. |
| `ctx.dry_run` | `microscope.dry_run`, for a plugin that must label what it did not do. |

A loop that can be stopped:

```python
for index, well in enumerate(wells):
    ctx.raise_if_stopped()
    scan_well(microscope, well)
    ctx.save_checkpoint(ScanProgress(done=index + 1))
    ctx.progress(index + 1, len(wells), well)
```

`StopRequested` derives from `Exception` and from nothing in
`smc.plugins.errors` or `smc.hardware.errors`, so an `except PluginError` or
`except SafetyRefusedError` in your loop never swallows a stop.

### Registering it

A plugin with an algorithm is a package, `smc/plugins/<snake_name>/`, with
`algorithm.py` and `plugin.py`. Register the **class** in `pyproject.toml`:

```toml
[project.entry-points."smc.plugins"]
stage-report = "smc.plugins.stage_report:StageReport"
```

The registry instantiates it with no argument. A third-party package registers
under the same group. Entry-point metadata is written at install time, so a plugin
that was added by editing `pyproject.toml` or by `git pull` is not listed until smc
is reinstalled: `pip install -e .` in the checkout (on a microscope PC, in its
`.venv`), or `python scripts/dev/worktree.py setup` in a worktree. Without it,
`smc run --list` says no plugin is registered.

A plugin that fails any check (it does not import, the name differs from the
entry point's, a member is missing, `requires` names something that is not a
capability) is not lost silently: `smc run --list` prints it with the reason,
and the others still load.

## The run document

One JSON file per experiment, ASCII only, written atomically. A stage's record is
keyed by the plugin's **name**; its result fills the **slot** the plugin provides.

```json
{
  "schema_version": 1,
  "created_utc": "2026-10-26T13:20:40.222054Z",
  "stages": {
    "stage-report": {
      "plugin": "stage-report",
      "provides": "stage-report",
      "attempt": "2cde6cd1db174a29bc51a4fdf396d259",
      "status": "done",
      "started_utc": "2026-10-26T13:20:40.222031Z",
      "finished_utc": "2026-10-26T13:20:40.222475Z",
      "profile": "demo",
      "dry_run": false,
      "smc_version": "0.0.1",
      "plugin_version": "0.0.1",
      "params": { "samples": 3, "interval_ms": 0.0, "include_z": true },
      "result": { "assumed": [], "profile": "demo", "readings": [] },
      "checkpoint": null,
      "reads": {},
      "error": null
    }
  }
}
```

- **Status.** `running` is written before `run` starts, so a crash shows
  `running`; then `done`, `stopped` (a stop, Ctrl-C included) or `failed`. Only
  `done` is a result to read: the others, `running` included, are never mistaken
  for one.
- **Attempts.** Every run of a stage has a new `attempt`, and a re-run replaces
  the stage's record when it starts. A stage that reads a slot records the
  `plugin` and `attempt` it read.
- **Slots.** The provider of a slot is the record with the latest start, even if
  it failed: a re-run of plate registration that failed leaves no usable plate,
  rather than the older one silently.
- **Staleness.** A stage is stale when something it read has been produced again
  since, directly or through a stage it read. `smc run` warns about the stages a
  run made stale.
- **A slot can be read** when it has a provider that ended `done`, is not stale,
  ran on the same profile, and was not a dry run (a dry run may read a real
  result). When it cannot, the error says why and what to run.
- **Nothing identifies a person or a PC**: no host name, user name or serial
  number, because run documents get attached to issues.
- **An unreadable document is never overwritten.** A corrupt file, a file written
  by a newer smc (`written by a newer smc (schema N); this smc reads schema 1`) or
  a file with a field this smc does not know is refused with its reason and left
  as it is: it may be the only record of an experiment. A byte-order mark from
  Notepad is accepted. A write that Windows refuses for a moment (antivirus, an
  indexer) is retried; if it still fails, the new text is kept in a
  `.<name>.<token>.smc-tmp` file next to the document and the error names it.
- **One writer at a time.** `write_stage` reads the file again before it writes,
  so a stage written meanwhile survives, but nothing locks the file.

## Testing a plugin

Test a plugin through `run_plugin`, with a run document under `tmp_path`, never by
building a context:

```python
def test_stage_report_reads_xy_and_z(demo_microscope, tmp_path):
    record = run_plugin(
        StageReport(),
        demo_microscope,
        StageReportParams(samples=3),
        tmp_path / "run.json",
    )
    assert record.status is StageStatus.done
```

- `fake_microscope` (a `FakeCore`, no Micro-Manager) is the fast bed for runner and
  logic tests; `demo_microscope` runs on Micro-Manager's demo devices
  (`@pytest.mark.demo`). Never `-m hardware` from a development machine.
- Test plugins are classes of the test module, loaded with
  `EntryPoint(name="x", value=f"{__name__}:Class", group="smc.plugins")` and passed to
  `discover(...)`/`load(...)`. Never put one under `src/` or in `smc.testing`.
- A test module's basename must be unique under `tests/`: pytest imports it by that
  name, and the entry point's `value` names the same module.
- Assert the text the code wrote (an error message, a record's `error`), not a word
  that the path or the fixture could also supply, and make sure each test can fail:
  break the line it guards and watch it fail.
- Test the pure algorithm on synthetic data and the loop on the demo devices
  ([ADR-0004](../adr/0004-plugin-architecture.md), [ADR-0005](../adr/0005-testing-strategy.md)).

`/new-plugin <name>` scaffolds all of this.
