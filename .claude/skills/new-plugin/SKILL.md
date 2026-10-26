---
name: new-plugin
description: Scaffold a new smc plugin (an interchangeable microscope tool) with the pure-algorithm / hardware-loop split, pydantic Params and Result, entry-point registration, demo-device tests and a docs page. Argument: the plugin name in kebab-case.
---

# /new-plugin <name>

Read ADR-0004 and `docs/design/m3-plugins.md` §1 first, then `docs/plugins/README.md`.
A plugin depends on **capabilities only** (ADR-0003). `<name>` is kebab-case
(`plate-find`); `<snake_name>` is the same in snake_case (`plate_find`), because a
kebab name is not a module path.

Create:

```
src/smc/plugins/<snake_name>/
  __init__.py      empty, or exports the plugin class
  algorithm.py     pure functions: arrays/numbers in, numbers out — no hardware
  plugin.py        Params, Result, and the class with run(microscope, params, ctx)
tests/plugins/test_<snake_name>_algorithm.py   unit + hypothesis on synthetic data
tests/plugins/test_<snake_name>_plugin.py      through run_plugin, on the demo devices
docs/plugins/<name>.md                         what it does, when to use it, its outputs
```

Rules:
- `Params` derives from `smc.plugins.PluginParams` and `Result` from
  `smc.plugins.PluginResult`: the bases make every float finite, refuse unknown
  keys and freeze the model. Bound every field and give it a `description`
  (`smc run <name> --show-params` prints them). Unknown is `None`, never `nan`.
- The class has `name`, `summary`, `Params`, `Result`, `requires`, `provides`,
  `reads`, `reads_optional` and `run`. `requires = (XYStage, Camera, …)` declares
  the capabilities; the runner checks them before `run`, and names every missing one
  with nothing written. `provides` is the slot of the run document the result fills
  (by convention the plugin's name); `reads` are the slots the runner insists on,
  `reads_optional` the ones used when usable; the two are disjoint and neither holds
  `provides`. Annotate the attributes (`requires: tuple[type, ...] = (XYStage,)`) so
  that mypy does not narrow them.
- `run` is the hardware loop. It checks `ctx.raise_if_stopped()` between units of
  work, reports through `ctx.progress()`, waits with `ctx.wait_s()` (**never
  `time.sleep`**: it cannot be stopped, and one long wait keeps Ctrl-C from being
  delivered on Windows), saves `ctx.save_checkpoint()` after work it would not
  repeat, reads upstream results through `ctx.upstream()` / `ctx.upstream_or_none()`,
  logs through `ctx.logger`, and never prints. It returns its `Result` and never
  writes the run document.
- Outputs that describe *where to image* are `useq` objects. Anything measured vs
  assumed is labelled in the `Result` (`assumed`).
- Register the **class** in `pyproject.toml` under
  `[project.entry-points."smc.plugins"]` as
  `<name> = "smc.plugins.<snake_name>.plugin:<ClassName>"`, then reinstall
  (`python scripts/dev/worktree.py setup` in a worktree): entry-point metadata is
  written at install time. Check `smc run --list` and `smc run <name> --show-params`.
- Test through `run_plugin` with a run document under `tmp_path`, on
  `fake_microscope` (no Micro-Manager) or `demo_microscope` (`@pytest.mark.demo`);
  never build a `RunContext`. A test plugin is a class of the test module, loaded
  with `EntryPoint(name=…, value=f"{__name__}:<Class>", group="smc.plugins")`.
  **Test module basenames are unique under `tests/`** (pytest imports them by
  basename, and an entry point names the same module): `test_<snake_name>_plugin.py`
  must not exist anywhere else.

If porting from `nikon-control`, keep the original module docstring's reasoning and
the original tests; adapt names to capabilities.
