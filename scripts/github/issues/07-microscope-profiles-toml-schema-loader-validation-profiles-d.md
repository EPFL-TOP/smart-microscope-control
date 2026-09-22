---
title: feat(core): microscope profiles (TOML) — schema, loader, validation, profiles/demo.toml
labels: [type: feature, area: core, priority: p0]
milestone: M1 — Stage on the simulator
---
## Goal
One TOML file per instrument is the only place where stand-specific facts live: config path, role overrides, safety limits, per-objective pixel sizes, quirks (ADR-0003). `profiles/demo.toml` sketches the shape.

## Acceptance criteria
- [ ] `smc.hardware.profile.Profile` (pydantic) with `load(name_or_path)`; `profiles/` searched by name; `tomllib`/`tomli`
- [ ] Validation errors name the file, the key and the fix
- [ ] `smc doctor --profile demo` and `smc profiles` (list) work
- [ ] Schema documented in `profiles/README.md`
