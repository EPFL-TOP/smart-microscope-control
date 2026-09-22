---
title: feat(core): ObjectiveTurret, LightSource and Channels capabilities
labels: [type: feature, area: core, priority: p1]
milestone: M2 — Nikon on hardware
---
## Goal
The remaining capabilities from ADR-0003: confirmed turret moves; the "how bright" knob found by property hints on light-like devices; Micro-Manager config-group presets as channels (only groups actually named for channels — never a `Camera` or `Objective` group).

## Context
`nikon-control` `Scope.{set_objective,intensity_property,channel_group,set_channel}`; `scope/channels.py` (capture the current illumination as a preset, exclude stage/turret/focus devices).

## Acceptance criteria
- [ ] `ObjectiveTurret.select(label, *, confirm=True)` raises without confirmation; per-objective pixel size from the profile
- [ ] `LightSource` reports which property it drives; absent knob is reported, not faked
- [ ] `Channels.capture(name)` writes a preset to the running core and to the `.cfg`, replacing a same-named one
- [ ] Simulator tests (demo has `DObjective`, `DWheel`, a shutter) + FakeCore for exclusions
