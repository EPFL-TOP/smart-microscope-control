---
title: feat(ui): web UI skeleton per ADR-0006 — connect, drive, snap/live, run a plugin
labels: [type: feature, area: ui, status: blocked, priority: p1]
milestone: M5 — Web UI
---
## Goal
The first graphical interface over the capability layer, built to the five principles of ADR-0006: one interface for every microscope; radically simple; a fixed shell that scales with plugins; safety visible; the UI is a view. Blocked until ADR-0006 is accepted and the framework spike has been decided.

## The shell (ADR-0006)
Connect (choose a microscope) → status strip with **Stop** → live view → drive (jog, Z, autofocus, objective, light) → tools rail grouped by *Set up · Find · Acquire · Watch* → one tool panel with a form generated from `Params`, Run/Stop, progress, result card → run document as a checklist. No other regions; a plugin cannot add a kind of screen.

## Acceptance criteria
- [ ] `[ui]` extra; `smc ui --profile P --port N`; deployable as a Windows service
- [ ] Identical screens on `demo` and on a real profile; missing capabilities shown disabled with the reason
- [ ] Live view ≥ 5 fps on the demo camera; Stop interrupts a running plugin via `ctx`
- [ ] Tests drive the UI without a browser against the demo devices
- [ ] A newcomer registers a plate on the demo devices without reading a manual (tested with a lab member)
