---
name: new-plugin
description: Scaffold a new smc plugin (an interchangeable microscope tool) with the pure-algorithm / hardware-loop split, pydantic Params and Result, entry-point registration, demo-device tests and a docs page. Argument: the plugin name in kebab-case.
---

# /new-plugin <name>

Read ADR-0004 first. A plugin depends on **capabilities only** (ADR-0003).

Create:

```
src/smc/plugins/<name>/
  __init__.py      exports Plugin
  algorithm.py     pure functions: arrays/numbers in, numbers out — no hardware
  plugin.py        Params (pydantic), Result (pydantic), Plugin.run(microscope, params, ctx)
tests/plugins/test_<name>_algorithm.py     unit + hypothesis on synthetic data
tests/plugins/test_<name>_plugin.py        runs on demo_microscope end to end
docs/plugins/<name>.md                     what it does, when to use it, its outputs
```

Rules:
- `requires = (XYStage, Camera, …)` declares the capabilities; the registry
  checks them before `run`.
- The hardware loop yields or checks `ctx.should_stop()` between steps and
  reports progress via `ctx.progress()`; never prints.
- Outputs that describe *where to image* are `useq` objects.
- Register in `pyproject.toml` under `[project.entry-points."smc.plugins"]`
  as `<name> = "smc.plugins.<name>:Plugin"` and check `smc run --list`.
- Anything measured vs assumed is labelled in the `Result`.

If porting from `nikon-control`, keep the original module docstring's
reasoning and the original tests; adapt names to capabilities.
