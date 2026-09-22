---
title: feat(plugins): plugin API — Plugin protocol, entry-point registry, Params/Result, run document, smc run
labels: [type: feature, area: plugins, priority: p0]
milestone: M3 — Plate tools as plugins
---
## Goal
Tools become interchangeable units (ADR-0004): a `Plugin` with `name`, `Params` (pydantic), `requires` (capabilities), `run(microscope, params, ctx) -> Result`; discovered through the `smc.plugins` entry-point group; reading/writing one JSON run document per experiment.

## Acceptance criteria
- [ ] `smc.plugins.Plugin` protocol, `registry.discover()`, `ctx` with `progress()`, `should_stop()`, logger
- [ ] `smc.run.RunDocument`: versioned JSON, stage outputs keyed by plugin name, atomic writes, resume
- [ ] `smc run --list`; `smc run <name> --profile P --run-doc run.json --param key=value` with params derived from the model
- [ ] A trivial built-in plugin (`stage-report`) exercises the whole path on the demo devices
- [ ] `requires` checked before `run`; readable error when a capability is missing
