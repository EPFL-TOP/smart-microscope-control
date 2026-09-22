---
title: feat(core): role resolution — profile overrides, core slots, type-first heuristics
labels: [type: feature, area: core, priority: p0]
milestone: M1 — Stage on the simulator
---
## Goal
Given a loaded core, decide which device fills each role — `camera`, `xy_stage`, `focus`, `autofocus`, `autofocus_offset`, `objective_turret`, `shutter`, `light_source`, `light_path`, `filter_turret` — and *report the candidates* so a wrong pick is visible.

## Context
Port and generalise `nikon-control/src/nikon_control/scope/stand.py` (`resolve_roles`, `role_choices`, `ambiguous_roles`, exclusion lists such as the Ti2 TIRF positioners typed `XYStage`). Order of authority: profile > MMCore role slots > heuristics (ADR-0003).

## Acceptance criteria
- [ ] `smc.hardware.roles.resolve(core, profile) -> RoleMap` with `.candidates`, `.ambiguous`, `.missing`
- [ ] A config that names an excluded device for a role is corrected and warned about (the `TIRF1`-as-XY case)
- [ ] Pure-Python resolver testable without a core (objects with `.name`/`.type`), plus a demo-devices test
- [ ] `smc devices` prints the role table with candidates
