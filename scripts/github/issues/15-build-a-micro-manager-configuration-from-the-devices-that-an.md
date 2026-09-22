---
title: feat(cli): build a Micro-Manager configuration from the devices that answer (smc config build)
labels: [type: feature, area: cli, scope: nikon-ti2, scope: nikon-tie, priority: p1]
milestone: M2 — Nikon on hardware
---
## Goal
Bring a rig up without the Java wizard: load the hub, ask it what is attached, load peripherals one at a time keeping those that initialise, add a camera, resolve roles, write the `.cfg` — carrying over channel presets and pixel sizes from an existing file.

## Context
Port of `nikon-control/src/nikon_control/scope/config_build.py` (+ `PRESERVED_DIRECTIVES`, `safe_label`). Only meaningful on a machine with real adapters; testable with the demo hub.

## Acceptance criteria
- [ ] `smc config build --profile X --out MMConfig.cfg [--camera-adapter …] [--skip …] [--dry-run] [--force]`
- [ ] Re-running never deletes `ConfigGroup`/`PixelSize` lines; `.bak` written on `--force`
- [ ] Hub failure prints the causes in order of likelihood (another program owns the stand; power order; DLL)
- [ ] Demo-devices test builds a config from `DemoCamera/DHub`
