---
title: feat(core): Microscope facade over CMMCorePlus with safety policy and dry-run
labels: [type: feature, area: core, priority: p0]
milestone: M1 — Stage on the simulator
---
## Goal
`Microscope.open(profile)` → roles resolved → `require(XYStage)` returns a capability. Safety guards live here and cannot be bypassed by a tool: jog guard, soft Z limits, Z↔autofocus interlock (M2), confirmed turret moves (M2), **dry-run** that logs instead of moving.

## Context
`nikon-control` `Scope` (control.py): `state()` reads everything tolerant of one broken device; `close()` releases the stand because only one connection exists. Device timeout 60 s (a plate traverse exceeds MMCore's 5 s default).

## Acceptance criteria
- [ ] `Microscope` context manager: open/close, `require()`, `has()`, `state()`, `describe()`
- [ ] `dry_run=True` makes every mutating call log and return the *commanded* value
- [ ] `CapabilityMissing` names the role and lists what is available
- [ ] Simulator tests for open/close/require/state; FakeCore test for the timeout being set
