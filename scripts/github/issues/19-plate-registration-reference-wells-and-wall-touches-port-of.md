---
title: feat(plugins): plate registration — reference wells and wall touches (port of plate.py)
labels: [type: feature, area: plugins, priority: p0]
milestone: M3 — Plate tools as plugins
---
## Goal
Register a well plate against the stage: fit `a1_center_xy` + `rotation` (scale fixed by the `useq` plate definition) from centred wells or from wall touches; refuse a wrong plate type (spacing off by > 2 %); output a `useq.WellPlatePlan` into the run document.

## Context
Port `nikon-control/src/nikon_control/scope/plate.py` with `tests/test_plate.py` (31 tests): `calibrate`, `centre_from_edges`, `seed`, `well_position`, `wall_position`, `nearest_well`, `suggested_refs`, layouts, persistence.

## Acceptance criteria
- [ ] `smc.plugins.plate_registration.algorithm` pure; hypothesis test: calibration round-trips through `useq`'s forward model
- [ ] Plugin steps: `seed`, `goto-well`, `goto-wall`, `capture-edge`, `calibrate` (CLI-drivable, UI-ready)
- [ ] Guided loop from `docs/scope-bringup.md §7` documented in `docs/plugins/plate-registration.md`
