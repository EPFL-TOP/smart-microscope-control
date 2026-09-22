---
title: feat(ui): web UI skeleton per ADR-0006 — connect, drive, snap/live, run a plugin
labels: [type: feature, area: ui, status: blocked, priority: p1]
milestone: M5 — Web UI
---
## Goal
The first graphical interface over the capability layer (blocked until ADR-0006 is accepted): choose a profile and connect; stage jog/goto with the guard; snap and live view; run any registered plugin from a form generated from its `Params`; show the run document as the pipeline state.

## Context
Re-expresses `nikon-control`'s `/scope` dashboard (Connect · Drive · Plate · Channels · Throughput tabs) over the new layer. Deployable as a service on a microscope PC (NSSM pattern from `docs/windows-deploy.md`).

## Acceptance criteria
- [ ] `[ui]` extra; `smc ui --profile P --port N`
- [ ] Live view ≥ 5 fps on the demo camera; stop button interrupts a running plugin via `ctx`
- [ ] Browser-driven test of the document against the demo devices (as `nikon-control` did with Bokeh)
