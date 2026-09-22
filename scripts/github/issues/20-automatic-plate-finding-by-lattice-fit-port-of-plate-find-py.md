---
title: feat(plugins): automatic plate finding by lattice fit (port of plate_find.py)
labels: [type: feature, area: plugins, priority: p1]
milestone: M3 — Plate tools as plugins
---
## Goal
Survey the stage on a jittered coarse grid (one brightness number per point, any objective), fit every plate type in `useq`'s registry, report type + `a1_center_xy` + rotation + score + runner-up margin. Measured on synthetic plates: correct type every time, 0.3–0.7 mm on a 96-well plate.

## Context
Port `nikon-control/src/nikon_control/scope/plate_find.py` (+ `tests/test_plate_find.py`): four-stage search (Fourier rotation candidates → phase sharpening on the infinite lattice → A1 node search both directions → local refinement); defaults chosen by measurement (step = pitch/3, margin = 1 pitch, jitter 0.5). Needs the synthetic sample issue for end-to-end tests.

## Acceptance criteria
- [ ] Pure `fit()`/`decide()` with the original synthetic-plate tests
- [ ] Hardware loop `survey()` over `XYStage` + `Camera`, stoppable, progress reported
- [ ] End-to-end demo test with the synthetic plate sample
